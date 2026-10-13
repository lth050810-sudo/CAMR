# CAMR V5 K8

This repository contains only the code snapshot for the CAMR V5 K8 method.

- `prepare.py` prepares and freezes the experiment inputs.
- `train.py` launches the eight-GPU training workflow.
- `evaluate.py` performs the final evaluation.
- `supervise.py` supervises preparation, training, and evaluation.
- `source_snapshot/baselines/` contains the CAMR core implementation and its required protocol utilities.

Models, datasets, checkpoints, logs, and outputs are not included.
