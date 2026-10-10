"""Plot completed, audited 3K trajectories with paradigm-specific panels."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'runs/wikibigedit_3k_qwen3_unknown'
BATCH = ROOT / 'runs/baseline_expansion_20260920'
PANELS = [
    ('Parameter learning', [
        ('cumulative_replay', 'Cumulative (higher compute)'),
        ('mir_lora_alllinear_r16_fair', 'MIR'), ('er_reservoir', 'ER'),
        ('relation_oracle_replay', 'Relation oracle'),
        ('semantic_replay_er_admission', 'Semantic (ER admission)'),
        ('tic_style_replay', 'TiC-style'),
        ('olora_shared_r16_alllinear', 'O-LoRA (growing rank, transfer)')]),
    ('Parameter editors: Qwen3 transfers', [
        ('memit_full100k_layers4_8_batch100_3k', 'MEMIT 4-8'),
        ('memit_nas_layers4_8_batch100_3k', 'MEMIT + NAS 4-8'),
        ('nas_transfer_layers7_11_batch100_3k', 'NAS 7-11'),
        ('alphaedit_full100k_layers5_8_batch100_3k', 'AlphaEdit 5-8 (sensitivity)'),
        ('rome_qwen25_transfer_layer5', 'ROME 5'),
        ('r_rome_qwen25_transfer_layer5', 'R-ROME 5')]),
    ('Side memory: Qwen3 transfers', [
        ('wise_retrieve_transfer_layer30', 'WISE'),
        ('horen_qwen25_transfer_layer24', 'HoReN')]),
    ('Open-book retrieval', [
        ('rag_mpnet_top2_paper_recipe', 'MPNet top-2 (paper recipe transfer)'),
        ('rag_dense_mpnet_top2', 'MPNet top-2 (shared prompt)'),
        ('rag_bm25_top2', 'BM25 top-2'),
        ('rag_hybrid_rrf_top2', 'Hybrid RRF top-2')]),
]


def main():
    out = BATCH / 'review'
    out.mkdir(exist_ok=True)
    audit = json.loads((BATCH / 'completion_audit.json').read_text())
    sources, omitted = {}, []
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True, sharey=True)
    for ax, (title, methods) in zip(axes.flat, PANELS):
        for name, label in methods:
            run = RUNS / name
            if not (run / 'completed.json').exists():
                omitted.append(name)
                continue
            if name in audit:
                assert audit[name]['status'] == 'verified_complete_3k', name
            metrics_path = run / 'metrics.json'
            metrics = json.loads(metrics_path.read_text())
            metrics = [m for m in metrics if m.get('current_stage', 0) in range(1, 9)]
            stages = [m['current_stage'] for m in metrics]
            assert stages in (list(range(1, 9)), [8]), name
            assert [m['num_facts'] for m in metrics] == [375 * s for s in stages], name
            final_only = stages == [8]
            ax.plot([m['num_facts'] for m in metrics],
                    [100 * m['update']['match_accuracy'] for m in metrics],
                    marker='o', markersize=6 if final_only else 3, linewidth=1.5,
                    markerfacecolor='none' if final_only else 'auto',
                    label=label + (' (final only)' if final_only else ''))
            sources[name] = {'metrics': str(metrics_path.relative_to(ROOT)),
                             'sha256': hashlib.sha256(metrics_path.read_bytes()).hexdigest(),
                             'evaluated_stages': stages}
        ax.set_title(title)
        ax.set_ylim(-2, 102)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8, loc='best')
        ax.set_xticks([375, 750, 1500, 2250, 3000])
    for ax in axes[1]:
        ax.set_xlabel('Facts seen')
    for ax in axes[:, 0]:
        ax.set_ylabel('Update accuracy on all facts seen (%)')
    fig.suptitle('WikiBigEdit / Qwen3-8B-Base: frozen 3K protocol')
    fig.text(.5, .012,
             'Single seed. Panels separate paradigms; compute and memory budgets differ. '
             + ('Partial report: unfinished runs omitted.' if omitted else 'All selected runs completed.'),
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .035, 1, .97))
    for extension in ('png', 'pdf'):
        fig.savefig(out / f'update_trajectories.{extension}', dpi=180)
    plt.close(fig)
    (out / 'plot_sources.json').write_text(json.dumps(
        {'sources': sources, 'omitted_unfinished': omitted}, indent=2) + '\n')
    print(f'Plotted {len(sources)} complete runs; {len(omitted)} unfinished runs omitted.')


if __name__ == '__main__':
    main()
