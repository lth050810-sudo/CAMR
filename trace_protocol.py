from baselines.common import qa_prompt
CONFIG = {'WikiBigEdit': (512, 512, 64, 128)}
def trim_prompt(ids, limit, task):
    if len(ids) > limit:
        raise ValueError('Unexpected prompt truncation')
    return ids
def training_encoding(tokenizer, row, task):
    prompt = tokenizer(qa_prompt(row['update']), add_special_tokens=True).input_ids
    answer = tokenizer(' ' + row['ans'], add_special_tokens=False).input_ids + [tokenizer.eos_token_id]
    if len(prompt) + len(answer) > 512:
        raise ValueError('Unexpected training truncation')
    return prompt + answer, len(prompt), len(answer)

def loss_coefficients(counts, facts, loss, current_weight, has_current):
    """Coefficient of each view's *summed* CE in the complete logical batch."""
    if not counts or min(counts) <= 0 or len(counts) != len(facts):
        raise ValueError('Invalid view token counts')
    if loss == 'token_mean':
        return [1.0 / sum(counts)] * len(counts)
    n = max(facts) + 1
    if set(facts) != set(range(n)) or loss != 'role_mean':
        raise ValueError('Invalid fact mapping or loss')
    view_counts = [facts.count(i) for i in range(n)]
    masses = ([1 / n] * n if not has_current else [1.0] if n == 1 else
              [current_weight] + [(1 - current_weight) / (n - 1)] * (n - 1))
    return [masses[f] / view_counts[f] / c for c, f in zip(counts, facts)]
