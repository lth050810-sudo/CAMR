"""Atomic stage barriers for the two missing-editor transfer runners."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from baselines.common import atomic_json, sha256_file


FILENAME = "editor_resume.pt"


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return value


def resume_identity(args, manifest_sha256, model_identity, provenance, source_files):
    runtime = {"gpu", "resume", "allow_existing_output"}
    recipe = {key: str(value.resolve()) if isinstance(value, Path) else value
              for key, value in vars(args).items() if key not in runtime}
    root = Path(__file__).resolve().parents[1]
    sources = [*source_files, "baselines/editor_checkpoint.py", "baselines/common.py",
               "baselines/run_lora_family.py", "baselines/wikibigedit_protocol.py"]
    weights = {path.name: {"bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
               for path in sorted(args.model_dir.glob("*.safetensors"))}
    return {"format": "editor-stage-identity/v1", "recipe": recipe,
            "manifest_sha256": manifest_sha256, "model_identity": model_identity,
            "weight_file_identity": weights, "statistics_provenance": provenance,
            "source_sha256": {name: sha256_file(root / name) for name in sources}}


def prediction_prefix(path, size=None):
    if not path.exists():
        if size not in (None, 0):
            raise ValueError("Committed predictions are missing")
        return 0, hashlib.sha256(b"").hexdigest()
    if size is None:
        size = path.stat().st_size
    if path.stat().st_size < size:
        raise ValueError("Predictions are shorter than the committed stage")
    digest = hashlib.sha256()
    remaining = size
    with path.open("rb") as stream:
        while remaining:
            block = stream.read(min(1 << 20, remaining))
            if not block:
                raise ValueError("Unexpected end of committed predictions")
            digest.update(block)
            remaining -= len(block)
    return size, digest.hexdigest()


def read_checkpoint(directory):
    path = directory / FILENAME
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != "editor-stage-checkpoint/v1":
        raise ValueError("Unsupported editor stage checkpoint")
    return payload


def save_stage(directory, parameters, optimizer, identity, *, cursor, seen_count,
               stage, metrics, training_log):
    size, digest = prediction_prefix(directory / "predictions.jsonl")
    cuda_device = next((value.device for value in parameters.values() if value.is_cuda), None)
    payload = {"format": "editor-stage-checkpoint/v1", "identity": identity,
               "cursor": cursor, "stage": stage, "seen_count": seen_count,
               "weights": {str(key): cpu_tree(value) for key, value in parameters.items()},
               "optimizer": cpu_tree(optimizer.state_dict()),
               "metrics": metrics, "training_log": training_log,
               "prediction_bytes": size, "prediction_prefix_sha256": digest,
               "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                       "torch": torch.get_rng_state(),
                       "cuda": torch.cuda.get_rng_state(cuda_device) if cuda_device is not None else None}}
    if hasattr(optimizer, "transforms"):
        payload["transforms"] = {str(key): cpu_tree(optimizer.transforms[value]) for key, value in parameters.items()}
    target = directory / FILENAME
    tmp = target.with_name(target.name + f".{os.getpid()}.tmp")
    torch.save(payload, tmp)
    tmp.replace(target)
    atomic_json(directory / "editor_resume.json", {"status": "stage_committed", "cursor": cursor,
        "stage": stage, "seen_count": seen_count, "prediction_bytes": size,
        "checkpoint_bytes": target.stat().st_size, "identity_sha256": hashlib.sha256(
            json.dumps(identity, sort_keys=True, default=str).encode()).hexdigest()})


def restore_stage(directory, parameters, optimizer, identity, *, payload=None):
    payload = read_checkpoint(directory) if payload is None else payload
    if payload["identity"] != identity:
        raise ValueError("Editor resume identity changed: model, manifest, recipe, statistics, or source")
    if set(payload["weights"]) != {str(key) for key in parameters}:
        raise ValueError("Checkpoint edited-layer set mismatch")
    for key, parameter in parameters.items():
        weight = payload["weights"][str(key)]
        if weight.shape != parameter.shape or weight.dtype != parameter.dtype or not torch.isfinite(weight).all():
            raise ValueError("Invalid edited weight in checkpoint")
    pred = directory / "predictions.jsonl"
    _, digest = prediction_prefix(pred, payload["prediction_bytes"])
    if digest != payload["prediction_prefix_sha256"]:
        raise ValueError("Committed prediction contents changed")
    with torch.no_grad():
        for key, parameter in parameters.items():
            parameter.copy_(payload["weights"][str(key)].to(parameter.device))
    optimizer.load_state_dict(payload["optimizer"])
    if hasattr(optimizer, "set_transforms"):
        optimizer.set_transforms({parameter: payload["transforms"][str(key)]
                                  for key, parameter in parameters.items()})
    if pred.exists():
        with pred.open("r+b") as stream:
            stream.truncate(payload["prediction_bytes"])
    atomic_json(directory / "metrics.json", payload["metrics"])
    atomic_json(directory / "training_log.json", payload["training_log"])
    random.setstate(payload["rng"]["python"])
    np.random.set_state(payload["rng"]["numpy"])
    torch.set_rng_state(payload["rng"]["torch"])
    if payload["rng"]["cuda"] is not None:
        device = next(value.device for value in parameters.values() if value.is_cuda)
        torch.cuda.set_rng_state(payload["rng"]["cuda"], device)
    with (directory / "resume_history.jsonl").open("a") as stream:
        stream.write(json.dumps({"at": time.time(), "cursor": payload["cursor"],
            "stage": payload["stage"], "prediction_bytes": payload["prediction_bytes"],
            "pid": os.getpid()}) + "\n")
    return payload
