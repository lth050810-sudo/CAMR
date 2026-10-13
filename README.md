# CAMR V5 K8

This repository contains only the latest CAMR V5 K8 code snapshot.

- `prepare.py` prepares the experiment inputs.
- `train.py` launches the eight-GPU training workflow.
- `evaluate.py` performs the final evaluation.
- `supervise.py` supervises preparation, training, and evaluation.
- `source_snapshot/baselines/` contains the CAMR core implementation and its required protocol utilities.

Models, datasets, checkpoints, logs, and outputs are not included.
