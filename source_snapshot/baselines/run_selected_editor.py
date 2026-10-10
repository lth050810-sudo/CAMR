"""Derive the final 3K command from a validated development certificate."""
import argparse
import json
from pathlib import Path
import sys

from baselines.common import atomic_json, sha256_file
from baselines.run_editor_job import run_spec


def make_spec(selection, certificate_path, manifest, output_dir):
    chosen = selection["selected"]
    method = selection["method"]
    modules = {"Adam-NSCL-Qwen3-transfer": "baselines.run_adam_nscl",
               "CrispEdit-Qwen3-transfer": "baselines.run_crispedit"}
    if method not in modules:
        raise ValueError("Unsupported selected editor")
    command = [sys.executable, "-u", "-m", modules[method], "--model-dir", chosen["model_dir"],
               "--manifest", str(manifest.resolve()), "--output-dir", str(output_dir.resolve()),
               "--stats-dir", chosen["stats_dir"], "--layers", ",".join(map(str, chosen["layers"])),
               "--gpu", "{gpu}", "--dev-selection", str(certificate_path.resolve()), "--eval-batch-size", "48"]
    for key, value in chosen["settings"].items():
        command.extend(["--" + key.replace("_", "-"), str(value)])
    if method == "CrispEdit-Qwen3-transfer":
        command.extend(["--chunk-size", "375", "--eval-every-chunks", "1", "--skip-base-eval"])
    return {"command": command, "output_dir": str(output_dir.resolve()), "cwd": "/root/engram",
            "checkpoint_dir": str(Path("/mnt/yy2607004_hs/engram_checkpoints") / output_dir.name),
            "selection_sha256": sha256_file(certificate_path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()
    selection = json.loads(args.selection.read_text())
    if selection["test_manifest_sha256"] != sha256_file(args.manifest):
        raise ValueError("Selection certificate belongs to a different test manifest")
    spec = make_spec(selection, args.selection, args.manifest, args.output_dir)
    # Keep this outside the output directory: the editor requires a fresh path.
    atomic_json(args.output_dir.parent / (args.output_dir.name + ".selected_command.json"), spec)
    raise SystemExit(run_spec(spec, args.gpu))


if __name__ == "__main__":
    main()
