"""Gate, freeze, and run the authorized six-method full WikiBigEdit phase.

The default only checks prerequisites. --prepare freezes CPU plans without
starting jobs. --run waits durably, then launches the shared leased-GPU queue.
"""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import string
import subprocess
import time
import unicodedata


ROOT = Path('/root/engram')
BASE = ROOT / 'runs/baseline_completion_20260920'
DEFAULT_CONFIG = BASE / 'full/full_phase_config.json'
TERMINAL_FAILURE = {'failed', 'blocked_dependency'}
SOURCES = ('__init__.py', 'common.py', 'wikibigedit_protocol.py', 'run_lora_family.py',
           'run_full_replay.py', 'run_rag.py', 'run_full_pre_edit_reference.py',
           'run_baseline_queue.py', 'summarize_runs.py', 'full_phase_controller.py')
FULL_METHODS = (
    ('rag_mpnet_top2_paper_recipe', 'dense_rag', 'rag'),
    ('lora_ft_block375', 'lora_ft', 'training'),
    ('rag_bm25_top2', 'bm25_rag', 'rag'),
    ('rag_hybrid_rrf_top2', 'hybrid_rag', 'rag'),
    ('er_reservoir_block375', 'er_reservoir', 'training'),
    ('relation_oracle_replay_block375', 'relation_oracle_replay', 'training'),
)


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            result.update(block)
    return result.hexdigest()


def object_sha256(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def immutable_json(path, value):
    path = Path(path)
    if path.exists():
        if object_sha256(json.loads(path.read_text())) != object_sha256(value):
            raise ValueError(f'Frozen artifact differs; use a new full phase directory: {path}')
    else:
        atomic_json(path, value)


def queue_gate(state_path, plan_path=None, *, optional=False):
    state_path = Path(state_path)
    if not state_path.exists():
        unscheduled = optional and not (plan_path and Path(plan_path).exists())
        return {'status': 'not_scheduled' if unscheduled else 'waiting_state',
                'ready': unscheduled, 'blocked': False, 'path': str(state_path)}
    state = json.loads(state_path.read_text())
    jobs = state.get('jobs')
    if not isinstance(jobs, dict) or not jobs:
        raise ValueError(f'Queue state has no job records: {state_path}')
    if plan_path and Path(plan_path).exists():
        plan = json.loads(Path(plan_path).read_text())
        names = [job['name'] for job in plan['jobs']]
        if state.get('plan_sha256') != sha256(plan_path) or set(jobs) != set(names):
            raise ValueError(f'Queue state and frozen plan disagree: {state_path}')
    failures = {name: {'status': job.get('status'), 'error': job.get('error'),
                       'exit_code': job.get('exit_code'), 'last_interruption': job.get('last_interruption')}
                for name, job in jobs.items() if job.get('status') in TERMINAL_FAILURE}
    complete = state.get('status') == 'complete' and all(j.get('status') == 'complete' for j in jobs.values())
    if state.get('status') == 'complete' and not complete:
        raise ValueError(f'Queue claims complete with incomplete jobs: {state_path}')
    blocked = bool(failures) or state.get('status') == 'failed'
    if complete and any(not isinstance(job.get('certificate'), dict) for job in jobs.values()):
        raise ValueError(f'Completed queue lacks output validation certificates: {state_path}')
    return {'status': state.get('status', 'unknown'), 'ready': complete, 'blocked': blocked,
            'path': str(state_path), 'job_counts': dict(Counter(j.get('status', 'unknown') for j in jobs.values())),
            'failures': failures, 'validated_completed_jobs': [name for name, j in jobs.items() if j.get('status') == 'complete'],
            'updated_at': state.get('updated_at')}


def prerequisite_gates(config):
    gates = {}
    for name, optional in (('data_screening', False), ('runner_smoke', False), ('baseline_3k', True)):
        settings = config['prerequisites'][name]
        try:
            gates[name] = queue_gate(settings['state'], settings.get('plan'), optional=optional)
        except (OSError, ValueError, TypeError, KeyError) as error:
            gates[name] = {'status': 'invalid_prerequisite', 'ready': False, 'blocked': True,
                           'path': settings['state'], 'error': str(error)}
    return {'ready': all(gate['ready'] for gate in gates.values()),
            'blocked': any(gate['blocked'] for gate in gates.values()), 'phases': gates,
            'waiting_for': [name for name, gate in gates.items() if not gate['ready']]}


def validate_screen_attachment(config):
    protocol = Path(config['protocol_dir'])
    summary_path = protocol / 'screen_attachment_summary.json'
    manifest_path = protocol / 'full_new_qwen3_unknown.json'
    summary = json.loads(summary_path.read_text())
    if summary.get('status') != 'complete' or Path(summary.get('manifest', '')).resolve() != manifest_path.resolve():
        raise ValueError('Full screen attachment summary is incomplete or points to another manifest')
    actual_hash = sha256(manifest_path)
    if summary.get('manifest_sha256') != actual_hash:
        raise ValueError('Attached full unknown manifest hash mismatch')
    manifest = json.loads(manifest_path.read_text())
    selection = manifest.get('selection', {})
    ids = selection.get('sample_ids')
    records = manifest.get('records')
    if not isinstance(ids, list) or not ids or len(ids) != len(set(ids)) or not isinstance(records, list):
        raise ValueError('Full unknown manifest needs a nonempty unique selection')
    record_map = {row['sample_id']: row for row in records}
    if len(record_map) != len(records) or not set(ids).issubset(record_map):
        raise ValueError('Full unknown selected/candidate IDs are inconsistent')
    selected = [record_map[sid] for sid in ids]
    screening = manifest.get('screening', {})
    statuses = dict(Counter(row.get('base_screen', {}).get('status') for row in records))
    if (screening.get('status') != 'complete' or screening.get('accepted_statuses') != ['unknown']
            or screening.get('applied_to_selection') is not True
            or screening.get('candidate_count') != len(records)
            or screening.get('screened_candidate_count') != len(records)
            or screening.get('missing_candidate_count') != 0
            or any(row.get('base_screen', {}).get('status') != 'unknown' for row in selected)
            or set(statuses) - {'known', 'unknown'}
            or statuses != screening.get('status_counts') or statuses != summary.get('status_counts')):
        raise ValueError('Full manifest is not a completely screened unknown-only selection')
    if (summary.get('candidate_count') != len(records) or summary.get('selected_unknown') != len(ids)
            or selection.get('total') != len(ids)):
        raise ValueError('Full attachment summary and selected counts disagree')
    stage_counts = dict(sorted(Counter(str(row['stage']) for row in selected).items()))
    if set(stage_counts) != set(map(str, range(1, 9))) or stage_counts != summary.get('stage_counts'):
        raise ValueError('Full selection must preserve all eight original chronological stages')
    if (manifest.get('protocol_scope') != 'full_eligible_new_fact_stream_dev_disjoint'
            or manifest.get('sampling', {}).get('allowed_tags') != ['new']):
        raise ValueError('Full phase requires the authorized complete eligible new-fact stream')
    post = manifest.get('postscreen_exclusions', {})
    excluded = set(post.get('train_candidate_eligibility_excluded_ids', []))
    if (post.get('screen_input_count_unchanged') != len(records)
            or post.get('train_candidate_eligibility_excluded_count') != len(excluded)
            or set(ids) & excluded or not excluded.issubset(record_map)):
        raise ValueError('Post-screen generation-budget exclusions are not bound to the selection')
    expected_unknown = {sid for sid, row in record_map.items()
                        if row['base_screen']['status'] == 'unknown' and sid not in excluded}
    if set(ids) != expected_unknown:
        raise ValueError('Final selection does not contain all eligible observed unknown facts')
    panel_counts = {}
    for axis in ('update', 'rephrase', 'personas', 'locality', 'mhop'):
        panel = manifest['panels'][axis]
        panel_ids = panel['sample_ids']
        if (len(panel_ids) != len(set(panel_ids)) or not set(panel_ids).issubset(ids)
                or panel.get('count') != len(panel_ids)
                or any(record_map[sid].get('panel_audit', {}).get(axis, {}).get('available') is not True for sid in panel_ids)):
            raise ValueError(f'Full manifest panel availability/coverage mismatch: {axis}')
        panel_counts[axis] = len(panel_ids)
    if panel_counts != summary.get('panels'):
        raise ValueError('Full attachment panel counts disagree')
    merged = protocol / 'qwen3_full_candidate_screen.jsonl'
    if not merged.exists() or sha256(merged) != screening.get('results_sha256'):
        raise ValueError('Full merged screen output is missing or changed')
    screen_state = json.loads(Path(config['prerequisites']['data_screening']['state']).read_text())
    attach_state = screen_state['jobs'].get('attach_full_unknown')
    if (not attach_state or attach_state.get('status') != 'complete'
            or attach_state.get('certificate', {}).get('sha256') != sha256(summary_path)):
        raise ValueError('Screen queue attachment certificate is missing or stale')
    return {'manifest': str(manifest_path), 'manifest_sha256': actual_hash,
            'summary': str(summary_path), 'summary_sha256': sha256(summary_path),
            'candidate_screen_count': len(records), 'selected_unknown_count': len(ids),
            'stage_counts': stage_counts, 'available_manifest_panel_counts': panel_counts,
            'generation_budget_excluded_candidates': len(excluded),
            'display_protocol_id': manifest.get('display_protocol_id'),
            'scope': 'all eligible screened unknown new facts; not all mixed-tag source records'}


def ensure_snapshot(config):
    destination = Path(config['output_root']) / '_controller_snapshot'
    expected = config['source_sha256']
    marker = destination / 'snapshot_manifest.json'
    if marker.exists():
        saved = json.loads(marker.read_text())
        if saved.get('source_sha256') != expected:
            raise ValueError('Existing full-phase source snapshot differs from this controller configuration')
    else:
        for relative, digest in expected.items():
            source = Path(config['repo_root']) / relative
            data = source.read_bytes()
            if hashlib.sha256(data).hexdigest() != digest:
                raise ValueError(f'Reviewed source changed before snapshot: {source}')
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and target.read_bytes() != data:
                raise ValueError(f'Partial source snapshot differs: {target}')
            if not target.exists():
                temporary = target.with_name(target.name + f'.{os.getpid()}.tmp')
                temporary.write_bytes(data)
                temporary.replace(target)
        atomic_json(marker, {'schema_version': 'full-phase-source-snapshot/v1',
                             'source_sha256': expected, 'created_at': time.time()})
    for relative, digest in expected.items():
        if sha256(destination / relative) != digest:
            raise ValueError(f'Frozen source snapshot changed: {relative}')
    return destination


def validate_smoke_source(config):
    settings = config['prerequisites']['runner_smoke']
    plan = json.loads(Path(settings['plan']).read_text())
    final_job = plan['jobs'][-1]
    output = Path(final_job['output'])
    done = json.loads(output.read_text())
    for field, expected in final_job.get('required_fields', {}).items():
        if done.get(field) != expected:
            raise ValueError(f'Full-runner smoke completion mismatch: {field}')
    state = json.loads(Path(settings['state']).read_text())
    certificate = state['jobs'][final_job['name']]['certificate']
    if certificate.get('sha256') != sha256(output):
        raise ValueError('Full-runner smoke completion certificate changed')
    recorded = json.loads((output.parent / 'run_config.json').read_text()).get('source_sha256', {})
    for name in ('run_full_replay.py', 'common.py', 'run_lora_family.py'):
        if recorded.get(name) != config['source_sha256']['baselines/' + name]:
            raise ValueError(f'Full-runner smoke did not validate this source version: {name}')
    return {'completion': str(output), 'completion_sha256': sha256(output),
            'source_sha256': recorded, 'scope': 'development smoke and resume check, not a full-data result'}


def training_command(config, method, manifest, output):
    return [config['python'], '-u', '-m', 'baselines.run_full_replay',
            '--method', method, '--manifest', str(manifest), '--output-dir', str(output),
            '--model-dir', config['model_dir'], '--gpu', '{gpu}', '--seed', '42',
            '--block-size', '375', '--memory-capacity', '1000', '--replay-ratio', '1.0',
            '--epochs', '5', '--batch-size', '8', '--gradient-accumulation-steps', '1',
            '--learning-rate', '0.0002', '--weight-decay', '0.0', '--max-grad-norm', '1.0',
            '--max-length', '128', '--max-new-tokens', '16',
            '--lora-r', '16', '--lora-alpha', '32', '--lora-dropout', '0.05',
            '--target-modules', 'q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj',
            '--panel-size-per-stage', '64', '--eval-batch-size', '48', '--checkpoint-every-blocks', '10']


def freeze_common_panel(config, snapshot, proof):
    directory = Path(config['output_root']) / '_shared_panel_plan'
    output = directory / 'plan.json'
    if not output.exists():
        directory.mkdir(parents=True, exist_ok=True)
        command = training_command(config, 'lora_ft', proof['manifest'], directory)
        command = [part.replace('{gpu}', '0') for part in command] + ['--plan-only']
        with (directory / 'freeze.log').open('a') as log:
            result = subprocess.run(command, cwd=snapshot, stdout=log, stderr=subprocess.STDOUT,
                                    env=dict(os.environ, CUDA_VISIBLE_DEVICES='', TOKENIZERS_PARALLELISM='false'), timeout=1800)
        if result.returncode:
            raise RuntimeError(f'CPU shared-panel freeze failed: exit {result.returncode}; see {directory / "freeze.log"}')
    plan = json.loads(output.read_text())
    panel = plan['evaluation_panels']
    settings = plan['config']
    if (settings.get('manifest_sha256') != proof['manifest_sha256']
            or settings.get('num_facts') != proof['selected_unknown_count']
            or settings.get('panel_size_per_stage') != 64 or settings.get('seed') != 42
            or settings.get('panel_fingerprint') != object_sha256(panel)):
        raise ValueError('Existing shared evaluation plan is not bound to this full unknown selection')
    return {'path': str(output), 'file_sha256': sha256(output),
            'panel_fingerprint': object_sha256(panel), 'counts': panel['counts'],
            'scope': 'one frozen sampled panel per axis shared by all six methods; not full-data five-axis evaluation'}


def build_full_jobs(config, snapshot, proof, panel):
    output_root = Path(config['output_root'])
    reference = output_root / 'pre_edit_reference'
    jobs = [{
        'name': 'pre_edit_reference', 'method': 'pre_edit_reference', 'phase': 'reference',
        'command': [config['python'], '-u', '-m', 'baselines.run_full_pre_edit_reference',
                    '--manifest', proof['manifest'], '--evaluation-panels', panel['path'],
                    '--output-dir', str(reference), '--model-dir', config['model_dir'], '--gpu', '{gpu}',
                    '--eval-batch-size', '48', '--max-length', '128', '--max-new-tokens', '16', '--resume'],
        'manifest': proof['manifest'], 'output': str(reference / 'completed.json'),
        'validation': 'json', 'required_fields': {
            'status': 'verified_complete', 'method': 'pre_edit_reference', 'reference_only': True,
            'manifest_sha256': proof['manifest_sha256'], 'panel_file_sha256': panel['file_sha256'],
            'panel_payload_sha256': panel['panel_fingerprint'],
            'evaluated_count': sum(panel['counts'].values()), 'source_selected_fact_count': proof['selected_unknown_count'],
            'parameter_updates': 0, 'context_fact_count': 0, 'scores_recomputed_from_predictions': True,
        }, 'gpu': True, 'cwd': str(snapshot), 'dependencies': [],
        'log': str(output_root / '_queue_logs/pre_edit_reference.log'),
        'common_panel_fingerprint': panel['panel_fingerprint'],
    }]
    for name, method, kind in FULL_METHODS:
        output = output_root / name
        if kind == 'training':
            command = training_command(config, method, proof['manifest'], output)
            validation = 'full_replay'
        else:
            command = [config['python'], '-u', '-m', 'baselines.run_rag',
                       '--method', method, '--recipe', 'wikibigedit' if method == 'dense_rag' else 'shared',
                       '--top-k', '2', '--manifest', proof['manifest'], '--evaluation-panels', panel['path'],
                       '--output-dir', str(output), '--model-dir', config['model_dir'], '--gpu', '{gpu}',
                       '--seed', '42', '--embedding-model', config['embedding_model'],
                       '--embedding-device', 'cpu', '--embedding-max-length', '384' if method == 'dense_rag' else '256',
                       '--embedding-cache-dir', str(output_root / '_rag_embedding_cache'),
                       '--reader-max-length', '512', '--max-new-tokens', '16',
                       '--retrieval-batch-size', '128', '--eval-batch-size', '32',
                       '--rrf-depth', '100', '--rrf-constant', '60', '--resume']
            validation = 'full_rag'
        jobs.append({'name': name, 'method': method, 'phase': kind, 'command': command,
                     'manifest': proof['manifest'], 'output': str(output / 'completed.json'),
                     'validation': validation, 'gpu': True, 'cwd': str(snapshot),
                     'resume_if_exists': kind == 'training', 'dependencies': ['pre_edit_reference'],
                     'log': str(output_root / '_queue_logs' / f'{name}.log'),
                     'common_panel_fingerprint': panel['panel_fingerprint']})
    return {'schema_version': 'baseline-experiment-queue/v1',
            'scope': 'authorized six-method full eligible unknown new-fact comparison plus sampled Pre-Edit reference',
            'selected_unknown_count': proof['selected_unknown_count'], 'manifest_sha256': proof['manifest_sha256'],
            'common_evaluation_panels': panel, 'source_snapshot': str(snapshot),
            'priority_note': 'Pre-Edit sampled reference first, then MPNet RAG and LoRA-FT; BM25 and Hybrid precede ER and Relation.',
            'jobs': jobs}


def prepare_full_phase(config):
    snapshot = ensure_snapshot(config)
    proof = validate_screen_attachment(config)
    smoke = validate_smoke_source(config)
    report_refresh = refresh_3k_summary(config, snapshot)
    panel = freeze_common_panel(config, snapshot, proof)
    plan = build_full_jobs(config, snapshot, proof, panel)
    output_root = Path(config['output_root'])
    plan_path = output_root / '_full_queue.json'
    immutable_json(plan_path, plan)
    preparation = {'status': 'prepared_not_launched', 'screen_attachment': proof,
                   'runner_smoke': smoke, 'shared_panel': panel, 'queue_plan': str(plan_path),
                   'baseline_3k_report_refresh': report_refresh,
                   'queue_plan_sha256': sha256(plan_path), 'snapshot': str(snapshot),
                   'prepared_method_names': [job['name'] for job in plan['jobs'] if job['phase'] != 'reference'],
                   'prepared_reference_names': ['pre_edit_reference']}
    immutable_json(output_root / '_prepared_phase.json', preparation)
    return preparation


def refresh_3k_summary(config, snapshot):
    editor_state = Path(config['prerequisites']['baseline_3k']['state'])
    if not editor_state.exists():
        return {'status': 'not_scheduled_editor_queue_absent'}
    editor = json.loads(editor_state.read_text())
    if editor.get('status') != 'complete' or any(job.get('status') != 'complete' for job in editor.get('jobs', {}).values()):
        raise ValueError('Cannot refresh completed 3K reports before the editor queue completes')
    directory = Path(config['output_root']) / '_3k_report_snapshot'
    marker = directory / 'refresh.json'
    if marker.exists():
        recorded = json.loads(marker.read_text())
        for artifact in recorded['artifacts'].values():
            if sha256(artifact['snapshot']) != artifact['sha256']:
                raise ValueError('Frozen 3K report snapshot changed')
        return recorded
    directory.mkdir(parents=True, exist_ok=True)
    runs = Path(config.get('baseline_runs_dir', ROOT / 'runs/wikibigedit_3k_qwen3_unknown'))
    command = [config['python'], '-m', 'baselines.summarize_runs', '--runs-dir', str(runs)]
    with (directory / 'refresh.log').open('a') as log:
        result = subprocess.run(command, cwd=snapshot, stdout=log, stderr=subprocess.STDOUT, timeout=1800)
    if result.returncode:
        raise RuntimeError(f'3K summary refresh failed with exit {result.returncode}')
    artifacts = {}
    for name in ('summary.json', 'summary.md'):
        source = runs / name
        data = source.read_bytes()
        target = directory / name
        temporary = target.with_name(target.name + f'.{os.getpid()}.tmp')
        temporary.write_bytes(data); temporary.replace(target)
        artifacts[name] = {'source': str(source), 'snapshot': str(target),
                           'sha256': hashlib.sha256(data).hexdigest()}
    evidence = {'status': 'refreshed_once', 'editor_queue_state': str(editor_state),
                'editor_queue_state_sha256': sha256(editor_state), 'command': command,
                'summarizer_source_sha256': config['source_sha256']['baselines/summarize_runs.py'],
                'artifacts': artifacts, 'refreshed_at': time.time()}
    atomic_json(marker, evidence)
    return evidence


def live_queue_process(state_path, expected_start=None):
    if not Path(state_path).exists():
        return None
    state = json.loads(Path(state_path).read_text())
    pid = state.get('pid')
    if not isinstance(pid, int) or state.get('status') in {'complete', 'failed'}:
        return None
    try:
        raw = (Path('/proc') / str(pid) / 'stat').read_text()
        start = raw[raw.rfind(')') + 2:].split()[19]
        argv = (Path('/proc') / str(pid) / 'cmdline').read_bytes().split(b'\0')
        argv = [value.decode() for value in argv if value]
        if 'baselines.run_baseline_queue' not in argv or '--state' not in argv:
            return None
        if Path(argv[argv.index('--state') + 1]).resolve() != Path(state_path).resolve():
            return None
        if expected_start is not None and str(expected_start) != start:
            return None
        return {'pid': pid, 'start_ticks': start}
    except (OSError, ValueError, IndexError):
        return None


def summarize_full_queue(config, preparation):
    path = Path(config['output_root']) / '_full_queue_state.json'
    if not path.exists():
        return {'status': 'not_started', 'completed_method_names': []}
    state = json.loads(path.read_text())
    plan = json.loads(Path(preparation['queue_plan']).read_text())
    if state.get('plan_sha256') != preparation['queue_plan_sha256'] or set(state.get('jobs', {})) != {j['name'] for j in plan['jobs']}:
        raise ValueError('Full experiment queue state does not match the prepared six-job plan')
    complete, references, failed, phases, locality = [], [], {}, {}, {}
    for job in plan['jobs']:
        status = state['jobs'][job['name']]
        phases[job['name']] = {key: status.get(key) for key in ('status', 'gpu', 'pid', 'attempts', 'log', 'error', 'certificate')}
        if status.get('status') == 'complete':
            if not isinstance(status.get('certificate'), dict):
                raise ValueError(f'Full method has no validation certificate: {job["name"]}')
            if sha256(job['output']) != status['certificate'].get('sha256'):
                raise ValueError(f'Full method completion changed after validation: {job["name"]}')
            cfg = json.loads((Path(job['output']).parent / 'run_config.json').read_text())
            observed = (cfg.get('panel_fingerprint') if job['phase'] == 'training'
                        else cfg.get('evaluation_panels_provenance', {}).get('panel_payload_sha256'))
            if observed != preparation['shared_panel']['panel_fingerprint']:
                raise ValueError(f'Full methods do not share the frozen panel: {job["name"]}')
            if job['phase'] == 'reference':
                references.append(job['name'])
            else:
                complete.append(job['name'])
                locality[job['name']] = locality_reference_comparison(config, job)
        elif status.get('status') in TERMINAL_FAILURE:
            failed[job['name']] = {'status': status['status'], 'error': status.get('error'), 'exit_code': status.get('exit_code')}
    result_status = state.get('status', 'unknown')
    if result_status == 'complete' and (len(complete) != len(FULL_METHODS) or references != ['pre_edit_reference']):
        raise ValueError('Full queue claims complete before all six methods and the reference validate')
    return {'status': result_status, 'completed_method_names': complete,
            'completed_reference_names': references, 'locality_vs_pre_edit': locality,
            'failed_methods': failed, 'jobs': phases, 'queue_state': str(path),
            'data_scope': preparation['screen_attachment'], 'evaluation_scope': preparation['shared_panel']}


def normalize_output(text):
    value = unicodedata.normalize('NFKC', str(text)).casefold()
    lines = value.splitlines()
    value = lines[0] if lines else value
    return ' '.join(''.join(' ' if character in string.punctuation else character for character in value).split())


def compare_locality_records(base, post):
    base_by_id = {record['sample_id']: record for record in base}
    post_by_id = {record['sample_id']: record for record in post}
    if len(base_by_id) != len(base) or len(post_by_id) != len(post) or not base:
        raise ValueError('Locality reference needs nonempty unique IDs')
    common = set(base_by_id) & set(post_by_id)
    for sid in common:
        if (base_by_id[sid]['question'], base_by_id[sid]['gold']) != (post_by_id[sid]['question'], post_by_id[sid]['gold']):
            raise ValueError('Locality reference question/answer differs from post-edit probe')
    raw = sum(base_by_id[sid]['prediction'] == post_by_id[sid]['prediction'] for sid in common)
    normalized = sum(normalize_output(base_by_id[sid]['prediction']) == normalize_output(post_by_id[sid]['prediction']) for sid in common)
    base_accuracy = sum(bool(record['match']) for record in base) / len(base)
    post_accuracy = sum(bool(record['match']) for record in post) / len(post) if post else None
    return {'base_count': len(base), 'post_count': len(post), 'paired_count': len(common),
            'coverage': len(common) / len(base), 'raw_equal_count': raw, 'raw_equal_accuracy': raw / len(base),
            'normalized_equal_count': normalized, 'normalized_equal_accuracy': normalized / len(base),
            'base_absolute_match_accuracy': base_accuracy, 'post_absolute_match_accuracy': post_accuracy,
            'match_delta_vs_pre_edit': post_accuracy - base_accuracy if post_accuracy is not None else None,
            'denominator_policy': 'all base locality records; missing post records fail'}


def locality_reference_comparison(config, job):
    reference = Path(config['output_root']) / 'pre_edit_reference'
    marker = json.loads((reference / 'completed.json').read_text())
    canonical = reference / 'predictions.jsonl'
    if marker.get('predictions_sha256') != sha256(canonical):
        raise ValueError('Pre-Edit reference predictions changed after completion')
    base = [json.loads(line) for line in canonical.read_text().splitlines() if line.strip()]
    base = [record for record in base if record['axis'] == 'locality']
    output = Path(job['output']).parent
    path = (output / 'evaluation/tau8/predictions.jsonl' if job['phase'] == 'training'
            else output / 'cache/evaluation/tau8/locality.jsonl')
    post = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    post = [record for record in post if record['axis'] == 'locality']
    comparison = compare_locality_records(base, post)
    if comparison['coverage'] != 1 or comparison['post_count'] != comparison['base_count']:
        raise ValueError('Completed full method lacks the exact shared locality reference panel')
    return comparison


def run_controller(config, state_path, mode='check'):
    state_path = Path(state_path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    lease = state_path.with_suffix('.lock').open('a+')
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    config_hash = object_sha256(config)
    state = json.loads(state_path.read_text()) if state_path.exists() else {
        'schema_version': 'full-phase-controller/v1', 'config_sha256': config_hash,
        'created_at': time.time(), 'completed_method_names': []}
    if state.get('config_sha256') != config_hash:
        lease.close()
        raise ValueError('Controller configuration changed; use a new controller state')
    events = state_path.with_suffix('.events.jsonl')
    stopping = False
    process = None
    queue_log = None

    def stop(*_):
        nonlocal stopping
        stopping = True

    def persist(status, phase, **updates):
        changed = (state.get('status'), state.get('phase')) != (status, phase)
        state.update(status=status, phase=phase, updated_at=time.time(), pid=os.getpid(), **updates)
        atomic_json(state_path, state)
        if changed:
            with events.open('a') as stream:
                stream.write(json.dumps({'at': state['updated_at'], 'status': status, 'phase': phase,
                                         'completed_method_names': state.get('completed_method_names', [])}) + '\n')
            print(json.dumps({'status': status, 'phase': phase, 'state': str(state_path)}), flush=True)

    previous_handlers = {sig: signal.signal(sig, stop) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        if mode != 'check':
            state['snapshot'] = str(ensure_snapshot(config))
        while not stopping:
            gates = prerequisite_gates(config)
            if gates['blocked']:
                persist('blocked_prerequisite', 'prerequisites', prerequisite_status=gates)
                return 1
            if not gates['ready']:
                persist('waiting_prerequisites', 'prerequisites', prerequisite_status=gates)
                if mode != 'run':
                    return 0
                time.sleep(config.get('poll_seconds', 10))
                continue
            if mode == 'check':
                persist('ready_to_prepare', 'prerequisites', prerequisite_status=gates)
                return 0
            if not state.get('preparation'):
                persist('preparing', 'freeze_full_protocol', prerequisite_status=gates)
                preparation = prepare_full_phase(config)
                state['preparation'] = preparation
            else:
                preparation = state['preparation']
                ensure_snapshot(config)
                if sha256(preparation['queue_plan']) != preparation['queue_plan_sha256']:
                    raise ValueError('Prepared queue plan changed')
            if mode == 'prepare':
                persist('prepared_not_launched', 'full_experiments', preparation=preparation)
                return 0
            summary = summarize_full_queue(config, preparation)
            if summary['status'] == 'complete':
                persist('complete', 'full_experiments', full_experiments=summary,
                        completed_method_names=summary['completed_method_names'])
                atomic_json(Path(config['output_root']) / '_full_phase_summary.json', state)
                return 0
            if summary['status'] == 'failed' or summary.get('failed_methods'):
                persist('blocked_full_job_failure', 'full_experiments', full_experiments=summary,
                        completed_method_names=summary['completed_method_names'])
                return 1
            queue_state = Path(config['output_root']) / '_full_queue_state.json'
            live = live_queue_process(queue_state)
            if process is not None and process.poll() is None:
                live = {'pid': process.pid}
            if live is None:
                if process is not None and process.poll() not in (None, 0):
                    raise RuntimeError(f'Full queue exited with code {process.returncode}; inspect {queue_state}')
                queue_log_path = Path(config['output_root']) / '_full_queue.log'
                queue_log_path.parent.mkdir(parents=True, exist_ok=True)
                queue_log = queue_log_path.open('a', buffering=1)
                command = [config['python'], '-u', '-m', 'baselines.run_baseline_queue',
                           '--plan', preparation['queue_plan'], '--state', str(queue_state),
                           '--gpus', '0,1,2,3,4,5,6,7', '--max-parallel', '2',
                           '--idle-seconds', '15', '--poll-seconds', '3']
                process = subprocess.Popen(command, cwd=preparation['snapshot'], stdout=queue_log,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                live = {'pid': process.pid, 'command': command, 'log': str(queue_log_path)}
            persist('running_full_queue', 'full_experiments', queue_process=live,
                    full_experiments=summary, completed_method_names=summary['completed_method_names'])
            time.sleep(config.get('poll_seconds', 10))
        persist('controller_interrupted', state.get('phase', 'prerequisites'),
                interruption_note='An already launched durable full queue keeps running; restart this controller to reattach.')
        return 130
    except Exception as error:
        persist('blocked_error', state.get('phase', 'prerequisites'), error=repr(error))
        return 1
    finally:
        for sig, previous in previous_handlers.items():
            signal.signal(sig, previous)
        if queue_log is not None:
            queue_log.close()
        lease.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--state', type=Path, default=BASE / 'full_phase_state.json')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--run', dest='mode', action='store_const', const='run')
    modes.add_argument('--prepare', dest='mode', action='store_const', const='prepare')
    parser.set_defaults(mode='check')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if not 0 < config.get('poll_seconds', 10) <= 60:
        parser.error('poll_seconds must be positive and no greater than 60')
    raise SystemExit(run_controller(config, args.state, args.mode))


if __name__ == '__main__':
    main()
