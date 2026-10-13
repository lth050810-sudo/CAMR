# CAMR V5 Paper Reference (K=4)

This repository contains the CAMR V5 code aligned with the paper's default
reference configuration. The K=8 replay-budget experiment is a scaling
variant in the paper; the primary configuration here uses K=4.

The default CAMR configuration enables safe `IGNORE` decisions. An ignored
arrival never trains the current fact; when safe history is available, its
maintenance step contains history only. For an active equivalent record with
no retirement request, `IGNORE` eligibility is determined solely by current
answer-token NLL <= 0.25, as specified in Section 4.2.

## Paper-reference configuration

| Parameter | Value |
|---|---:|
| Probe cap P | 4 |
| Similarity threshold | 0.65 |
| Mastery NLL threshold | 0.25 |
| Replay cap K | 4 |
| Writer projection dimension | 32 |
| Rollout length | 4 |
| Current-fact loss weight alpha | 0.5 |
| Retention reward weight | 1.0 |
| Token-cost weight | 0.01 |
| Cost reference C_ref | 128 |
| Discount gamma | 0.95 |
| Entropy coefficient | 0.01 |
| Critic coefficient | 0.5 |

ADD and IGNORE sample replay uniformly. MERGE reserves a uniform portion of
the K=4 budget and learns the remaining compatible targets. Reward uses the
raw current-NLL gain plus retention gain minus normalized token cost.

## Local assets

The repository contains no private paths. Set these variables before running:

```bash
export CAMR_DATA_ROOT=/path/to/WikiBigEdit/full_hf_viewer
export CAMR_MODEL_DIR=/path/to/Qwen3-8B-Base
export CAMR_ENCODER_DIR=/path/to/all-mpnet-base-v2
export CAMR_CHECKPOINT_ROOT=/path/to/camr_checkpoints
```

`CAMR_PYTHON` is optional; `supervise.py` otherwise uses the active Python
interpreter. `prepare.py` reads the committed source snapshot and never
rewrites training code.

- `prepare.py` prepares the experiment inputs.
- `train.py` launches the eight-GPU training workflow.
- `evaluate.py` performs the final evaluation.
- `supervise.py` supervises preparation, training, and evaluation.
- `source_snapshot/baselines/` contains the CAMR core implementation and its required protocol utilities.

Models, datasets, checkpoints, logs, and outputs are not included.
