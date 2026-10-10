"""Run an existing baseline over the raw stream with a shared sampled evaluation.

The adapter changes input ordering and evaluation selection only. Training
algorithms remain in their original modules. Missing auxiliary questions do
not remove a valid training QA. All changes are recorded in run_config.json.
"""
from __future__ import annotations

import argparse
from collections import Counter
import importlib
import json
import math
import os
from pathlib import Path
import sys

from baselines.run_full_replay import AXES, file_sha256, load_stream, object_sha256


RUNNERS = (
    'run_locft_bf', 'run_stableedit', 'run_ultraedit', 'run_memit',
    'run_alphaedit', 'run_rome_family', 'run_olora', 'run_wise',
    'run_nas_family', 'run_horen', 'run_crispedit', 'run_adam_nscl',
    'run_lora_merge', 'run_mir_lora', 'run_lora_family', 'run_full_ft_fsdp',
)


def read_panels(path, rows, metadata):
    payload = json.loads(Path(path).read_text())
    panels = payload.get('evaluation_panels', payload)
    mapping = {row['sample_id']: row for row in rows}
    for axis, fields in AXES.items():
        ids = panels['sample_ids'][axis]
        if not ids or len(ids) != len(set(ids)) or not set(ids).issubset(mapping):
            raise ValueError(f'invalid {axis} fixed panel')
        source_axis = {'persona': 'personas', 'multihop': 'mhop'}.get(axis, axis)
        allowed = set(metadata['panels'][source_axis]['sample_ids'])
        if not set(ids).issubset(allowed):
            raise ValueError(f'{axis} panel resurrects a masked question')
        if any(not isinstance(mapping[sid].get(field), str) or not mapping[sid][field].strip()
               for sid in ids for field in fields):
            raise ValueError(f'{axis} panel contains missing questions or answers')
    return panels


def filter_axis(rows, axis, panels):
    allowed = set(panels['sample_ids'][axis])
    # Intersect the supplied prefix. In particular, never introduce future
    # update/rephrase/persona facts by consulting the complete panel alone.
    return [row for row in rows if row['sample_id'] in allowed]


def training_coverage(directory, rows):
    """Verify trace identities where available; otherwise verify stage counts."""
    expected_ids = [row['sample_id'] for row in rows]
    expected_stages = dict(Counter(row['stage'] for row in rows))
    traces = directory / 'edit_trace.jsonl'
    if traces.exists():
        observed = []
        for line in traces.open():
            value = json.loads(line)
            if value.get('finite') is False:
                raise ValueError('nonfinite edit trace')
            if 'sample_ids' in value:
                observed.extend(value['sample_ids'])
            elif 'sample_id' in value:
                observed.append(value['sample_id'])
        if observed:
            if observed != expected_ids:
                raise ValueError('edit identities do not cover the complete source order')
            return {'evidence': 'edit_trace.jsonl ordered sample IDs', 'count': len(observed)}
    path = directory / 'training_log.json'
    if not path.exists():
        raise ValueError('missing full training coverage evidence')
    logs = json.loads(path.read_text())
    if not isinstance(logs, list):
        raise ValueError('training log is not a list')
    # CrispEdit uses bounded source-contiguous chunks rather than stages.
    if logs and all('chunk' in row and 'num_current_facts' in row for row in logs):
        offset = 0
        for number, value in enumerate(logs, 1):
            count = value['num_current_facts']
            selected = rows[offset:offset + count]
            if (value['chunk'] != number or len(selected) != count
                    or value.get('first_sample_id') != selected[0]['sample_id']
                    or value.get('last_sample_id') != selected[-1]['sample_id']):
                raise ValueError('chunk coverage is not a continuous source prefix')
            offset += count
        if offset != len(rows):
            raise ValueError('training chunks are incomplete')
        return {'evidence': 'chunk counts and boundary identities', 'count': offset}
    observed = {}
    for value in logs:
        stage = value.get('stage')
        if stage is None and str(value.get('scope', '')).startswith('tau'):
            stage = int(value['scope'][3:])
        count = value.get('current_count', value.get('num_current_facts', value.get('num_facts')))
        if stage is None or count is None or int(stage) in observed:
            raise ValueError('missing/duplicate training stage coverage')
        observed[int(stage)] = int(count)
        ids = value.get('sample_ids')
        if ids is not None and ids != [row['sample_id'] for row in rows if row['stage'] == int(stage)]:
            raise ValueError('training stage identities differ')
        steps = value.get('edit_steps')
        if isinstance(steps, list):
            actual = [sid for step in steps for sid in step.get('sample_ids', [])]
            expected = [row['sample_id'] for row in rows if row['stage'] == int(stage)]
            if actual != expected:
                raise ValueError('stage edit identities differ')
    if observed != expected_stages:
        raise ValueError(f'incomplete training stage coverage: {observed}')
    return {'evidence': 'per-stage training counts (IDs where logged)', 'count': sum(observed.values())}


def check_tensor_tree(value, label):
    """Check mapped tensors in bounded chunks, including large side memories."""
    import torch
    if isinstance(value, torch.Tensor):
        if value.numel() < 1:
            return 0
        flat = value.reshape(-1)
        for start in range(0, flat.numel(), 1 << 20):
            if not torch.isfinite(flat[start:start + (1 << 20)]).all():
                raise ValueError(f'nonfinite model tensor: {label}')
        return 1
    if isinstance(value, dict):
        return sum(check_tensor_tree(item, label + '.' + str(key)) for key, item in value.items())
    if isinstance(value, (tuple, list)):
        return sum(check_tensor_tree(item, label + '.' + str(index)) for index, item in enumerate(value))
    return 0


def verify_torch_artifact(path, *, tensor_key, expected_names=None, expected_format=None,
                          expected_stage=None, expected_base=None):
    import torch
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f'missing model artifact: {path}')
    try:
        payload = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    except Exception as error:
        raise ValueError(f'unreadable model artifact: {path}: {error}') from error
    if not isinstance(payload, dict) or tensor_key not in payload:
        raise ValueError(f'model artifact has no {tensor_key}: {path}')
    if expected_format and payload.get('format') != expected_format:
        raise ValueError(f'model artifact format mismatch: {path}')
    if expected_stage is not None and payload.get('stage') != expected_stage:
        raise ValueError(f'model artifact stage mismatch: {path}')
    if expected_base and payload.get('base_model') and Path(payload['base_model']).resolve() != Path(expected_base).resolve():
        raise ValueError(f'model artifact base model mismatch: {path}')
    tensors = payload[tensor_key]
    if expected_names is not None and (not isinstance(tensors, dict) or set(tensors) != set(expected_names)):
        raise ValueError(f'model artifact parameter names mismatch: {path}')
    count = check_tensor_tree(tensors, str(path))
    if not count:
        raise ValueError(f'model artifact contains no nonempty tensors: {path}')
    # WISE has one current matrix plus its stored retrieval memories.
    if tensor_key == 'new_weight':
        count += check_tensor_tree(payload.get('memory_weights', []), str(path) + '.memory_weights')
    if tensor_key == 'values':
        if not isinstance(payload.get('keys'), torch.Tensor):
            raise ValueError(f'HoReN artifact has no memory keys: {path}')
        count += check_tensor_tree(payload['keys'], str(path) + '.keys')
    return {'path': str(path.resolve()), 'bytes': path.stat().st_size,
            'sha256': file_sha256(path), 'finite_tensor_count': count}


def verify_fsdp_artifacts(directory, config, completed, final_stage):
    """Validate DCP metadata and all expected byte extents without loading weights."""
    from torch.distributed.checkpoint import FileSystemReader
    path = Path(completed.get('final_checkpoint', ''))
    root = Path(config['checkpoint_dir']).resolve()
    if not path.is_absolute() or path.resolve().parent != root:
        raise ValueError('FullFT final checkpoint is outside its configured checkpoint directory')
    path = path.resolve()
    meta_path = path / 'checkpoint_meta.json'
    try:
        meta = json.loads(meta_path.read_text())
    except (OSError, ValueError) as error:
        raise ValueError('missing or unreadable FullFT checkpoint metadata') from error
    expected = {'schema_version': 'engram-fsdp-checkpoint/v1', 'stage': final_stage,
                'manifest_sha256': config['manifest_sha256'], 'has_optimizer': bool(config.get('save_optimizer', True)),
                'world_size': config['world_size']}
    if any(meta.get(key) != value for key, value in expected.items()):
        raise ValueError('FullFT checkpoint metadata does not match the completed run')
    for key in ('model_dir', 'manifest'):
        if Path(meta.get(key, '')).resolve() != Path(config[key]).resolve():
            raise ValueError(f'FullFT checkpoint {key} mismatch')
    try:
        metadata = FileSystemReader(str(path)).read_metadata()
    except Exception as error:
        raise ValueError('missing or unreadable FullFT distributed checkpoint metadata') from error
    names = set(metadata.state_dict_metadata)
    if not any(name.startswith('model.') for name in names):
        raise ValueError('FullFT distributed checkpoint has no model state')
    if expected['has_optimizer'] and not any(name.startswith('optimizer.') for name in names):
        raise ValueError('FullFT distributed checkpoint has no optimizer state')
    index = Path(config['model_dir']) / 'model.safetensors.index.json'
    if index.exists():
        base_names = set(json.loads(index.read_text())['weight_map'])
        if not {'model.' + name for name in base_names}.issubset(names):
            raise ValueError('FullFT distributed checkpoint is missing base-model parameters')
    extents = {}
    if not metadata.storage_data:
        raise ValueError('FullFT distributed checkpoint has no shard extents')
    for storage in metadata.storage_data.values():
        relative = Path(storage.relative_path)
        if relative.is_absolute() or not (path / relative).resolve().is_relative_to(path):
            raise ValueError('FullFT shard path escapes its checkpoint directory')
        if storage.offset < 0 or storage.length <= 0:
            raise ValueError('FullFT shard has invalid byte extents')
        extents[str(relative)] = max(extents.get(str(relative), 0), storage.offset + storage.length)
    shards = []
    for relative, expected_bytes in sorted(extents.items()):
        shard = path / relative
        if not shard.is_file() or shard.stat().st_size != expected_bytes:
            raise ValueError(f'FullFT shard is missing or truncated: {relative}')
        shards.append({'path': str(shard), 'bytes': expected_bytes})
    return [{'path': str(path), 'checkpoint_meta_sha256': file_sha256(meta_path),
             'distributed_metadata_sha256': file_sha256(path / '.metadata'),
             'model_state_entries': sum(name.startswith('model.') for name in names),
             'shards': shards, 'validation': 'metadata bindings and exact shard byte extents; weights not loaded'}]


def model_artifacts(directory, config, completed, final_stage):
    runner = config['full_dataset_adapter']['runner']
    base = config.get('model_dir')
    if runner == 'run_full_ft_fsdp':
        return verify_fsdp_artifacts(directory, config, completed, final_stage)
    artifacts = []
    if runner == 'run_locft_bf':
        artifact = verify_torch_artifact(directory / 'locft_delta.pt', tensor_key='delta',
            expected_format='single_parameter_delta/v1', expected_base=base)
        import torch
        payload = torch.load(directory / 'locft_delta.pt', map_location='cpu', weights_only=True, mmap=True)
        if payload.get('parameter_name') != config.get('edited_parameter'):
            raise ValueError('LocFT saved parameter differs from the trained parameter')
        artifacts.append(artifact)
    elif runner == 'run_crispedit':
        artifacts.append(verify_torch_artifact(directory / 'crispedit_delta.pt', tensor_key='deltas',
            expected_names=config['edited_parameters'], expected_format='multi-parameter-delta/v1', expected_base=base))
    elif runner in {'run_stableedit', 'run_ultraedit', 'run_memit', 'run_alphaedit'}:
        if config.get('save_delta'):
            names = config.get('edit_modules')
            if runner in {'run_memit', 'run_alphaedit'}:
                names = [f'model.layers.{layer}.mlp.down_proj.weight' for layer in config['layers']]
            artifacts.append(verify_torch_artifact(config['save_delta'], tensor_key='delta',
                                                  expected_names=names, expected_base=base))
    elif runner == 'run_lora_merge':
        if config.get('save_edited_qv'):
            artifacts.append(verify_torch_artifact(directory / 'edited_qv_weights.pt', tensor_key='state_dict',
                expected_format='qwen3_qv_replacement/v1', expected_base=base))
    elif runner == 'run_wise':
        if config.get('save_adapter'):
            artifacts.append(verify_torch_artifact(config['save_adapter'], tensor_key='new_weight',
                expected_format='wise_retrieve_adapter/v1', expected_base=base))
    elif runner == 'run_olora':
        artifacts.append(verify_torch_artifact(directory / 'adapters/olora.pt', tensor_key='modules'))
    elif runner in {'run_rome_family', 'run_nas_family', 'run_adam_nscl', 'run_horen'}:
        # These original runners derive this path from the output-directory basename.
        root = Path('/mnt/yy2607004_hs/engram_checkpoints') / directory.name
        memory = runner == 'run_horen'
        path = root / f"tau{final_stage}_{'memory' if memory else 'delta'}.pt"
        artifacts.append(verify_torch_artifact(path, tensor_key='values' if memory else 'deltas',
                                              expected_stage=final_stage))
    elif runner in {'run_lora_family', 'run_mir_lora'}:
        from safetensors import safe_open
        root = directory / 'adapters' / f'tau{final_stage}'
        weights, adapter_config = root / 'adapter_model.safetensors', root / 'adapter_config.json'
        try:
            adapter = json.loads(adapter_config.read_text())
            if any(adapter.get(key) != config[key] for key in ('r', 'lora_alpha', 'lora_dropout') if key in config):
                raise ValueError('LoRA adapter configuration mismatch')
            if 'lora_r' in config and adapter.get('r') != config['lora_r']:
                raise ValueError('LoRA saved rank differs')
            with safe_open(str(weights), framework='pt', device='cpu') as saved:
                count = sum(check_tensor_tree(saved.get_tensor(key), key) for key in saved.keys())
            if not count:
                raise ValueError('LoRA adapter contains no weights')
        except Exception as error:
            raise ValueError(f'missing or invalid final LoRA adapter: {root}: {error}') from error
        artifacts.append({'path': str(weights), 'bytes': weights.stat().st_size, 'sha256': file_sha256(weights),
                          'config_sha256': file_sha256(adapter_config), 'finite_tensor_count': count})
    if runner in {'run_crispedit', 'run_adam_nscl'}:
        try:
            pointer = json.loads((directory / 'editor_resume.json').read_text())
            resume = directory / 'editor_resume.pt'
            if (pointer.get('status') != 'stage_committed' or pointer.get('stage') != final_stage
                    or pointer.get('seen_count') != config['num_facts']
                    or resume.stat().st_size != pointer['checkpoint_bytes']):
                raise ValueError('editor resume checkpoint is incomplete')
        except (OSError, KeyError, ValueError) as error:
            raise ValueError('missing or incomplete final editor resume checkpoint') from error
        artifacts.append({'path': str(resume), 'bytes': resume.stat().st_size,
                          'pointer_sha256': file_sha256(directory / 'editor_resume.json')})
    return artifacts


def validate_bindings(job, directory, config, info):
    runner = info['runner']
    if runner not in RUNNERS:
        raise ValueError('unknown adapter runner')
    for name, recorded in info['source_sha256'].items():
        if Path(name).name != name or file_sha256(Path(__file__).with_name(name)) != recorded:
            raise ValueError(f'adapter or runner source changed: {name}')
    if not {'run_full_baseline_adapter.py', runner + '.py', 'common.py'}.issubset(info['source_sha256']):
        raise ValueError('adapter source binding is incomplete')
    if Path(config['manifest']).resolve() != Path(info['manifest']).resolve():
        raise ValueError('runner and adapter manifest differ')
    if config.get('manifest_sha256') != info['manifest_sha256']:
        raise ValueError('runner and adapter manifest hashes differ')
    if Path(config['output_dir']).resolve() != directory.resolve():
        raise ValueError('runner output directory differs from validated output')
    if job.get('manifest') and Path(job['manifest']).resolve() != Path(info['manifest']).resolve():
        raise ValueError('queue manifest differs from adapter manifest')
    if job.get('command'):
        command = [str(value).replace('{gpu}', '0') for value in job['command']]
        start = command.index('baselines.run_full_baseline_adapter') + 1
        arguments = command[start:]
        separator = arguments.index('--')
        outer, forwarded = arguments[:separator], arguments[separator + 1:]
        for flag, expected in [('--runner', runner), ('--manifest', info['manifest']),
                               ('--evaluation-panels', info['evaluation_panels']), ('--output-dir', str(directory))]:
            index = outer.index(flag)
            if outer[index + 1] != expected:
                raise ValueError(f'queue adapter argument differs: {flag}')
        strip_runtime = lambda values: [value for value in values if value != '--resume']
        if strip_runtime(forwarded) != strip_runtime(info['forwarded_arguments']):
            raise ValueError('queue recipe differs from adapter forwarded arguments')
    expected_panel = job.get('common_panel_fingerprint')
    if expected_panel and expected_panel != info['panel_fingerprint']:
        raise ValueError('queue panel differs from adapter panel')


def validate_completed(job, output=None):
    from baselines.common import answer_scores
    output = Path(output or job['output'])
    directory = output.parent
    config = json.loads((directory / 'run_config.json').read_text())
    info = config['full_dataset_adapter']
    validate_bindings(job, directory, config, info)
    rows, metadata = load_stream(Path(info['manifest']))
    if file_sha256(Path(info['manifest'])) != info['manifest_sha256']:
        raise ValueError('raw manifest changed')
    panels = read_panels(info['evaluation_panels'], rows, metadata)
    if object_sha256(panels) != info['panel_fingerprint']:
        raise ValueError('evaluation panel changed')
    if config.get('num_facts') != len(rows):
        raise ValueError('runner did not select the complete training stream')
    if info.get('training_facts') != len(rows) or info.get('source_scope') != metadata.get('protocol_scope'):
        raise ValueError('adapter data scope/count does not match its manifest')
    if metadata.get('protocol_scope') == 'raw_all_source_records' and (len(rows) != 502382 or len(metadata['records']) != len(rows)):
        raise ValueError('raw-all adapter does not contain all 502382 source rows')
    completed = json.loads((directory / 'completed.json').read_text())
    if completed.get('status') != 'complete':
        raise ValueError('underlying runner is incomplete')
    coverage = training_coverage(directory, rows)
    artifacts = model_artifacts(directory, config, completed, max(row['stage'] for row in rows))
    metrics = json.loads((directory / 'metrics.json').read_text())
    final = metrics[-1]
    if final.get('current_stage') != max(row['stage'] for row in rows):
        raise ValueError('final stage was not evaluated')
    checkpoint = final['checkpoint']
    observed = {axis: {} for axis in AXES}
    for line in (directory / 'predictions.jsonl').open():
        record = json.loads(line)
        if record['checkpoint'] != checkpoint:
            continue
        axis, sid = record['axis'], record['sample_id']
        if axis not in AXES or sid in observed[axis]:
            raise ValueError('duplicate/invalid final prediction')
        score = answer_scores(record['prediction'], record['gold'])
        if any(record[key] != value for key, value in score.items()):
            raise ValueError('prediction scores do not recompute')
        observed[axis][sid] = record
    source_rows = {row['sample_id']: row for row in rows}
    for axis, (question, answer) in AXES.items():
        if set(observed[axis]) != set(panels['sample_ids'][axis]):
            raise ValueError(f'{axis} final prediction coverage differs')
        for sid, record in observed[axis].items():
            if record['question'] != source_rows[sid][question] or record['gold'] != source_rows[sid][answer]:
                raise ValueError('prediction question/gold changed from the manifest')
        count = len(observed[axis])
        for name in ('match', 'exact'):
            correct = sum(int(record[name]) for record in observed[axis].values())
            metric = final[axis]
            if (metric['count'] != count or metric[name + '_count'] != correct
                    or not math.isclose(metric[name + '_accuracy'], correct / count, abs_tol=1e-12)):
                raise ValueError('final metric does not recompute')
    return {'status': 'verified_complete', 'manifest_sha256': info['manifest_sha256'],
            'training_facts': len(rows), 'training_coverage': coverage,
            'panel_fingerprint': info['panel_fingerprint'], 'panel_counts': panels['counts'],
            'prediction_scores_recomputed': True, 'evaluation_scope': 'sampled panels',
            'model_artifacts': artifacts,
            'completion_sha256': file_sha256(directory / 'completed.json')}


def install_adapter(common, manifest, rows, metadata, panels, info):
    original_load = common.load_manifest
    original_axis = common.evaluate_axis
    original_atomic = common.atomic_json
    original_panel = common.manifest_panel
    by_id = {row['sample_id']: row for row in rows}

    def load(path):
        if Path(path).resolve() == manifest.resolve():
            return list(rows), metadata
        return original_load(path)

    def axis(model, tokenizer, selected, axis_name, device, **kwargs):
        return original_axis(model, tokenizer, filter_axis(selected, axis_name, panels), axis_name, device, **kwargs)

    def panel(payload, name, limit=0):
        if payload is metadata:
            axis_name = {'personas': 'persona', 'mhop': 'multihop'}.get(name, name)
            values = [by_id[sid] for sid in panels['sample_ids'][axis_name]]
            return values[:limit or None]
        return original_panel(payload, name, limit=limit)

    def atomic(path, value):
        if Path(path).name == 'run_config.json':
            value = dict(value, full_dataset_adapter=info,
                         evaluation_scope='shared fixed sampled panels; training covers raw source rows')
            value.setdefault('num_facts', len(rows))
        return original_atomic(path, value)

    common.load_manifest = load
    common.evaluate_axis = axis
    common.manifest_panel = panel
    common.atomic_json = atomic


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runner', choices=RUNNERS, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--evaluation-panels', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    forwarded = args.arguments[1:] if args.arguments[:1] == ['--'] else args.arguments
    if any(value.split('=')[0] in {'--max-stages', '--examples-per-stage', '--max-edits', '--allow-existing-output'}
           for value in forwarded):
        raise ValueError('full baseline adapter rejects subset/overwrite flags')
    if any(value.split('=')[0] in {'--manifest', '--output-dir'} for value in forwarded):
        raise ValueError('manifest/output are set by the adapter')
    rows, metadata = load_stream(args.manifest)
    panels = read_panels(args.evaluation_panels, rows, metadata)
    info = {'schema_version': 'raw-full-baseline-adapter/v1', 'runner': args.runner,
            'manifest': str(args.manifest.resolve()), 'manifest_sha256': file_sha256(args.manifest),
            'source_scope': metadata.get('protocol_scope'), 'training_facts': len(rows),
            'evaluation_panels': str(args.evaluation_panels.resolve()),
            'panel_fingerprint': object_sha256(panels), 'panel_counts': panels['counts'],
            'ordering': 'original stage then numeric source row index',
            'missing_auxiliary_questions': 'excluded from that evaluation axis only; training QA retained',
            'training_algorithm': 'unchanged original runner',
            'source_sha256': {name: file_sha256(Path(__file__).with_name(name))
                              for name in ('run_full_baseline_adapter.py', args.runner + '.py', 'common.py')},
            'forwarded_arguments': forwarded}
    if args.runner == 'run_ultraedit':
        info['source_sha256']['ultraedit_full_scale.py'] = file_sha256(Path(__file__).with_name('ultraedit_full_scale.py'))
        info['full_scale_execution'] = 'same primal ridge equations accumulated by token blocks; FP32 summation order differs; CPU feature lists released per module'
    from baselines import common
    install_adapter(common, args.manifest, rows, metadata, panels, info)
    module = importlib.import_module('baselines.' + args.runner)
    if args.runner == 'run_ultraedit':
        from baselines.ultraedit_full_scale import apply_ultraedit_stage
        module.apply_ultraedit_stage = apply_ultraedit_stage
    sys.argv = [module.__file__, '--manifest', str(args.manifest), '--output-dir', str(args.output_dir), *forwarded]
    module.main()
    if int(os.environ.get('RANK', '0')) == 0:
        certificate = validate_completed({'output': str(args.output_dir / 'full_completed.json')})
        common.atomic_json(args.output_dir / 'full_completed.json', certificate)


if __name__ == '__main__':
    main()
