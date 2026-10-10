"""Forward-KL SDFT primitives matching the authors' April 2026 correction."""
import torch
import torch.nn.functional as F
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from baselines.common import qa_prompt


def teacher_prompt(question, demonstration):
    # Base-model completion adaptation of the released demonstration template.
    return (f'Question: {question.strip()}\n\n'
            f'This is an example for a response to the question:\n{demonstration.strip()}\n\n'
            'Now answer the question with a short answer of your own.\nAnswer:')


def encode_prompts(tokenizer, rows, *, teacher=False, max_length=128):
    texts = [teacher_prompt(r['update'], r['ans']) if teacher else qa_prompt(r['update']) for r in rows]
    ids = [tokenizer(text, add_special_tokens=True).input_ids for text in texts]
    if any(len(tokens) > max_length for tokens in ids):
        raise ValueError('A prompt exceeds its declared limit; refusing to truncate its demonstration')
    width = max(map(len, ids))
    padded = torch.full((len(ids), width), tokenizer.pad_token_id, dtype=torch.long)
    mask = torch.zeros_like(padded)
    for index, tokens in enumerate(ids):
        padded[index, -len(tokens):] = torch.tensor(tokens)
        mask[index, -len(tokens):] = 1
    return {'input_ids': padded, 'attention_mask': mask}


def completion_logits(model, prompts, completion_ids, completion_mask):
    input_ids = torch.cat((prompts['input_ids'], completion_ids), -1)
    attention = torch.cat((prompts['attention_mask'], completion_mask), -1)
    position_ids = (attention.cumsum(-1) - 1).clamp(min=0)
    logits = model(input_ids=input_ids, attention_mask=attention, position_ids=position_ids,
                   logits_to_keep=completion_ids.shape[1] + 1, use_cache=False).logits
    return logits[:, :-1].float()


def forward_kl_per_sequence(student_logits, teacher_log_probs, completion_mask, skip_tokens=0):
    student_log_probs = F.log_softmax(student_logits.float(), -1)
    teacher_log_probs = teacher_log_probs.detach()
    per_token = F.kl_div(student_log_probs, teacher_log_probs,
                         reduction='none', log_target=True).sum(-1)
    mask = completion_mask.clone()
    if skip_tokens:
        mask[:, :skip_tokens] = 0
    # The released trainer averages over tokens within each sequence, then sequences.
    return (per_token * mask).sum(-1) / mask.sum(-1).clamp(min=1)


@torch.no_grad()
def update_ema(student, teacher, alpha):
    source, target = dict(student.named_parameters()), dict(teacher.named_parameters())
    if source.keys() != target.keys():
        raise ValueError('Student and EMA teacher parameter layouts differ')
    for name, parameter in source.items():
        teacher_parameter = target[name]
        if parameter.shape != teacher_parameter.shape or parameter.dtype != teacher_parameter.dtype:
            raise ValueError('Student and EMA teacher shards differ: ' + name)
        teacher_parameter.mul_(1 - alpha).add_(parameter.detach(), alpha=alpha)
    if isinstance(teacher, FSDP):
        # FSDP keeps the root's gathered weights after a forward, anticipating
        # backward. A frozen teacher has no backward to release that cache;
        # merely changing its local shards would leave embedding/head logits
        # using old weights. Invalidate any cached gathers after each EMA.
        from torch.distributed.fsdp._runtime_utils import _reshard
        for module in FSDP.fsdp_modules(teacher):
            handle = module._handle
            if handle is None or not handle.uses_sharded_strategy:
                continue
            gathered = getattr(handle.flat_param, '_full_param_padded', None)
            if gathered is not None and gathered.untyped_storage().nbytes():
                _reshard(module, handle, True)
