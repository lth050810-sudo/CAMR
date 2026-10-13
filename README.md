# CAMR

This repository contains the CAMR implementation described in the paper.

The default CAMR configuration enables safe `IGNORE` decisions. An ignored
arrival never trains the current fact; when safe history is available, its
maintenance step contains history only. For an active equivalent record with
no retirement request, `IGNORE` eligibility is determined solely by current
answer-token NLL <= 0.25, as specified in Section 4.2.

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
