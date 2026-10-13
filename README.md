# CAMR V5 K8 with IGNORE

This repository contains only the latest CAMR V5 K8 code snapshot, with the
ADD, MERGE, and IGNORE policy actions enabled. The default training entry point
uses the original IGNORE behavior: a certified redundant current fact is
skipped while selected safe history is maintained.

- `prepare.py` prepares and freezes the experiment inputs.
- `train.py` launches the eight-GPU training workflow.
- `evaluate.py` performs the final evaluation.
- `supervise.py` supervises preparation, training, and evaluation.
- `source_snapshot/baselines/` contains the CAMR core implementation and its required protocol utilities.

Models, datasets, checkpoints, logs, and outputs are not included.
