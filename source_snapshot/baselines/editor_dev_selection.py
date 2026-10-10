"""Freeze configurations from completed independent-dev editor runs.

Criterion is the unweighted mean of the five generated match accuracies at
the final development stage. All candidates must be listed before looking
at test outcomes; the selected configuration and its evidence are hash bound.
"""
import argparse
import json
import math
from collections.abc import Mapping
from pathlib import Path

from baselines.common import load_manifest, sha256_file
from baselines.wikibigedit_protocol import _normalized_key


AXES = ("update", "rephrase", "persona", "locality", "multihop")
SETTING_KEYS = ("seed", "epochs", "batch_size", "learning_rate", "weight_decay",
                "loss_threshold", "max_length", "max_new_tokens",
                "nullspace_threshold", "energy_threshold")

ROME_METHODS = frozenset({"ROME", "R-ROME"})
MEMIT_METHODS = frozenset({"MEMIT"})

# These are CLI/runtime settings, so a selected development configuration can
# be checked directly against the subsequent formal run.  Preset-only ROME
# values are recorded separately in ``hyperparameters`` below.
ROME_SETTING_KEYS = (
    "seed",
    "eval_batch_size",
    "max_length",
    "max_new_tokens",
    "loss_layer",
    "mom2_adjustment",
    "mom2_stats_dir",
    "mom2_ridge",
    "max_update_weight_ratio",
    "oversized_update_action",
    "max_cumulative_weight_drift_ratio",
)
REQUIRED_ROME_SETTING_KEYS = frozenset(ROME_SETTING_KEYS)
MEMIT_SETTING_KEYS = (
    "seed",
    "eval_batch_size",
    "max_length",
    "max_new_tokens",
    "v_loss_layer",
    "v_num_grad_steps",
    "v_lr",
    "v_weight_decay",
    "clamp_norm_factor",
    "kl_factor",
    "mom2_update_weight",
    "solver_ridge",
    "max_update_weight_ratio",
    "edit_batch_size",
    "representation_batch_size",
    "context_mode",
    "stats_source",
    "stats_dir",
    "stats_sample_size",
    "stats_file_template",
)
REQUIRED_MEMIT_SETTING_KEYS = frozenset(MEMIT_SETTING_KEYS)
REQUIRED_ROME_HPARAM_KEYS = frozenset({
    "fact_token",
    "v_num_grad_steps",
    "v_lr",
    "v_weight_decay",
    "clamp_norm_factor",
    "kl_factor",
    "context_template_length_params",
    "rewrite_module_tmp",
})
ROME_PRESET_EXCLUDED_KEYS = frozenset({
    # Bound elsewhere, or deliberately irrelevant to candidate equivalence.
    "device", "model_name", "layers", "v_loss_layer", "stats_dir",
})
PATH_SETTING_KEYS = frozenset({"mom2_stats_dir", "stats_dir"})
PROTOCOL_SETTING_KEYS = (
    "seed",
    "eval_batch_size",
    "max_length",
    "max_new_tokens",
    "max_edits",
    "max_stages",
    "examples_per_stage",
    "no_stage_eval",
    "locality_panel_size",
    "multihop_panel_size",
    "closed_book_evaluation",
    "evaluation_prompt_template",
)
FORMAL_STREAM_EDIT_COUNT = 3000
# MEMIT has no sequential trust-region cap, so its independent-dev selection
# must reject an abrupt quality collapse before comparing final scores.  These
# values are part of every new certificate's criterion/evidence.
MEMIT_QUALITY_DROP_THRESHOLD = 0.20
MEMIT_QUALITY_RATIO_THRESHOLD = 0.50


def _method_family(method):
    normalized = str(method).strip().upper().replace("_", "-")
    if normalized in ROME_METHODS:
        return "rome"
    if normalized in MEMIT_METHODS:
        return "memit"
    return "legacy"


def _canonical_method(method):
    normalized = str(method).strip().upper().replace("_", "-")
    if normalized in ROME_METHODS:
        return normalized
    if normalized in MEMIT_METHODS:
        return normalized
    return str(method)


def _normalize_setting(key, value):
    if value is not None and key in PATH_SETTING_KEYS:
        return str(Path(value).resolve())
    return value


def _canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _settings(config, method):
    family = _method_family(method)
    keys = (ROME_SETTING_KEYS if family == "rome" else
            MEMIT_SETTING_KEYS if family == "memit" else SETTING_KEYS)
    required = (REQUIRED_ROME_SETTING_KEYS if family == "rome" else
                REQUIRED_MEMIT_SETTING_KEYS if family == "memit" else frozenset())
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"Candidate is missing selection settings: {missing}")
    return {
        key: _normalize_setting(key, config[key])
        for key in keys
        if key in config
    }


def _protocol_settings(config):
    return {
        key: config[key]
        for key in PROTOCOL_SETTING_KEYS
        if key in config
    }


def _rome_hyperparameters(config):
    """Return the stable EasyEdit preset values that define a ROME variant."""

    upstream = config.get("upstream_hparams", {})
    if upstream is None:
        upstream = {}
    if not isinstance(upstream, Mapping):
        raise ValueError("ROME upstream_hparams must be an object")
    missing = sorted(REQUIRED_ROME_HPARAM_KEYS - upstream.keys())
    if missing:
        raise ValueError(f"ROME upstream_hparams is missing selection values: {missing}")
    return {
        key: value
        for key, value in sorted(upstream.items())
        if key not in ROME_PRESET_EXCLUDED_KEYS
    }


def _rome_mom2_adjustment(config):
    upstream = config.get("upstream_hparams", {}) or {}
    upstream_value = upstream.get("mom2_adjustment") if isinstance(upstream, Mapping) else None
    direct_value = config.get("mom2_adjustment")
    if direct_value is not None and upstream_value is not None and direct_value != upstream_value:
        raise ValueError("ROME mom2_adjustment disagrees with upstream_hparams")
    return direct_value if direct_value is not None else upstream_value


def _layers(config):
    layers = config.get("qwen3_layers", config.get("layers"))
    if (not isinstance(layers, list) or not layers or
            any(not isinstance(layer, int) or isinstance(layer, bool) for layer in layers) or
            len(layers) != len(set(layers)) or
            any(layer < 0 for layer in layers) or layers != sorted(layers)):
        raise ValueError("Candidate layers must be a non-empty sorted list of unique non-negative integers")
    return layers


def _loss_layer(config, method, layers):
    family = _method_family(method)
    if family == "rome":
        value = config.get("loss_layer")
        upstream = config.get("upstream_hparams", {}) or {}
        upstream_value = upstream.get("v_loss_layer") if isinstance(upstream, Mapping) else None
        if value is None:
            value = upstream_value
        elif upstream_value is not None and value != upstream_value:
            raise ValueError("ROME loss_layer disagrees with upstream_hparams.v_loss_layer")
        upstream_layers = upstream.get("layers") if isinstance(upstream, Mapping) else None
        if upstream_layers is not None and list(upstream_layers) != layers:
            raise ValueError("ROME layers disagree with upstream_hparams.layers")
    elif family == "memit":
        value = config.get("v_loss_layer")
    else:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError("Candidate loss layer must be an integer at or after every rewrite layer")
    invalid_order = value <= max(layers) if family == "rome" else value < max(layers)
    if invalid_order:
        raise ValueError("Candidate loss layer must be an integer at or after every rewrite layer")
    return value


def _provenance_entries(raw, label):
    if isinstance(raw, Mapping):
        items = []
        for layer, value in raw.items():
            if not isinstance(value, Mapping):
                raise ValueError(f"{label} entries must be objects")
            entry = dict(value)
            if "layer" not in entry:
                try:
                    entry["layer"] = int(layer)
                except (TypeError, ValueError):
                    pass
            items.append(entry)
        return items
    if isinstance(raw, list):
        if not all(isinstance(entry, Mapping) for entry in raw):
            raise ValueError(f"{label} entries must be objects")
        return [dict(entry) for entry in raw]
    raise ValueError(f"{label} must be a list or object")


def _validate_file_provenance(raw, label, layers=None):
    """Re-hash every external statistics artifact and return canonical evidence."""

    entries = _provenance_entries(raw, label)
    if not entries:
        raise ValueError(f"Candidate has no {label}")
    normalized = []
    for entry in entries:
        source = entry.get("path", entry.get("source"))
        expected = entry.get("sha256", entry.get("file_sha256"))
        if not source or not expected:
            raise ValueError(f"{label} entry requires path and file_sha256")
        path = Path(source).resolve()
        try:
            actual = sha256_file(path)
        except (FileNotFoundError, IsADirectoryError, PermissionError) as error:
            raise ValueError(f"Candidate {label} file is unavailable: {path}") from error
        if actual != expected:
            raise ValueError(f"Candidate {label} changed: {path}")
        item = {"path": str(path), "file_sha256": expected}
        if "layer" in entry:
            try:
                item["layer"] = int(entry["layer"])
            except (TypeError, ValueError) as error:
                raise ValueError(f"{label} layer must be an integer") from error
        normalized.append(item)
    normalized.sort(key=lambda item: (item.get("layer", -1), item["path"]))
    if layers is not None:
        recorded_layers = [item.get("layer") for item in normalized]
        if all(layer is None for layer in recorded_layers):
            if len(normalized) != len(layers):
                raise ValueError(f"Candidate {label} does not cover every selected layer")
        elif (any(layer is None for layer in recorded_layers) or
              set(recorded_layers) != set(layers) or len(recorded_layers) != len(layers)):
            raise ValueError(f"Candidate {label} does not cover exactly the selected layers")
    return normalized


def _require_development_markers(config, completed):
    if config.get("development_only") is not True or completed.get("development_only") is not True:
        raise ValueError("Selection candidates require development_only=true")
    if (config.get("formal_baseline_eligible") is not False or
            completed.get("formal_baseline_eligible") is not False):
        raise ValueError("Selection candidates require formal_baseline_eligible=false")


def _memit_quality_trajectory(metrics):
    """Audit the eight-stage MEMIT quality trajectory.

    ``quality_collapse`` is intentionally a pass flag for compatibility with
    the selection evidence contract: it is true when no adjacent-stage pair
    satisfies the collapse rule, and false when one does.  A collapse is a
    decrease strictly greater than 0.20 in the five-axis macro mean together
    with a next/previous ratio strictly below 0.50.
    """

    by_stage = []
    for record in metrics:
        try:
            stage = int(record["current_stage"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("MEMIT quality trajectory has an invalid stage") from error
        scores = []
        for axis in AXES:
            try:
                value = record[axis]["match_accuracy"]
            except (KeyError, TypeError) as error:
                raise ValueError(
                    f"MEMIT quality trajectory is missing {axis} at stage {stage}"
                ) from error
            if isinstance(value, bool):
                raise ValueError("MEMIT quality score must be numeric")
            try:
                value = float(value)
            except (TypeError, ValueError) as error:
                raise ValueError("MEMIT quality score must be numeric") from error
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("MEMIT quality score must be finite in [0, 1]")
            scores.append(value)
        by_stage.append((stage, sum(scores) / len(scores)))
    if [stage for stage, _ in by_stage] != list(range(1, 9)):
        raise ValueError("MEMIT quality trajectory requires stages 1 through 8")

    events = []
    for (previous_stage, previous), (stage, current) in zip(by_stage, by_stage[1:]):
        drop = previous - current
        ratio = current / previous if previous > 0 else 0.0
        if drop > MEMIT_QUALITY_DROP_THRESHOLD and ratio < MEMIT_QUALITY_RATIO_THRESHOLD:
            events.append({
                "from_stage": previous_stage,
                "to_stage": stage,
                "previous_macro": previous,
                "next_macro": current,
                "drop": drop,
                "next_over_previous": ratio,
            })
    return {
        # Historical naming retained per the selection protocol: true means
        # the trajectory passed the collapse gate; false means collapse.
        "quality_collapse": not events,
        "quality_macro_by_stage": {
            str(stage): macro for stage, macro in by_stage
        },
        "quality_collapse_events": events,
        "quality_gate": {
            "drop_threshold": MEMIT_QUALITY_DROP_THRESHOLD,
            "ratio_threshold": MEMIT_QUALITY_RATIO_THRESHOLD,
            "rule": (
                "reject when adjacent five-axis macro drop is strictly greater "
                "than drop_threshold and next/previous is strictly less than "
                "ratio_threshold"
            ),
        },
    }


def _rome_stream_capacity(path, config, completed, dev_count):
    """Audit whether a ROME trust region has capacity for the formal stream.

    Sequential rank-one updates accumulate approximately like a random walk,
    so the observed development drift is projected by sqrt(N_formal/N_dev).
    A candidate that scores well on the short development stream remains in
    the certificate as evidence, but cannot be selected when its configured
    cumulative cap is below that projection.
    """
    trace_path = path / "edit_trace.jsonl"
    try:
        lines = [line for line in trace_path.read_text().splitlines() if line.strip()]
    except OSError as error:
        raise ValueError(f"ROME candidate has no readable edit trace: {trace_path}") from error
    if len(lines) != dev_count:
        raise ValueError(
            f"ROME edit trace has {len(lines)} records, expected {dev_count}"
        )
    if completed.get("edits") != dev_count:
        raise ValueError("ROME completed edit count does not match the development manifest")
    if completed.get("skipped_edits") != 0:
        raise ValueError("ROME development candidate skipped one or more edits")

    max_observed = 0.0
    skipped = 0
    for index, line in enumerate(lines, start=1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"ROME edit trace record {index} is invalid JSON") from error
        if not isinstance(record, Mapping):
            raise ValueError(f"ROME edit trace record {index} must be an object")
        if record.get("finite") is not True:
            raise ValueError(f"ROME edit trace record {index} is not marked finite")
        if record.get("trust_region_skipped") is not False:
            skipped += 1
        ratios = record.get("cumulative_weight_drift_ratios")
        if not isinstance(ratios, Mapping) or not ratios:
            raise ValueError(
                f"ROME edit trace record {index} has no cumulative drift ratios"
            )
        for value in ratios.values():
            if isinstance(value, bool):
                raise ValueError("ROME cumulative drift ratio must be numeric")
            try:
                ratio = float(value)
            except (TypeError, ValueError) as error:
                raise ValueError("ROME cumulative drift ratio must be numeric") from error
            if not math.isfinite(ratio) or ratio < 0:
                raise ValueError("ROME cumulative drift ratio must be finite and non-negative")
            max_observed = max(max_observed, ratio)
    if skipped:
        raise ValueError("ROME development candidate skipped one or more edits")

    configured = config.get("max_cumulative_weight_drift_ratio")
    if isinstance(configured, bool):
        raise ValueError("ROME cumulative drift cap must be numeric")
    try:
        configured = float(configured)
    except (TypeError, ValueError) as error:
        raise ValueError("ROME cumulative drift cap must be numeric") from error
    if not math.isfinite(configured) or configured <= 0:
        raise ValueError("ROME cumulative drift cap must be finite and positive")
    projection_scale = math.sqrt(FORMAL_STREAM_EDIT_COUNT / dev_count)
    projected = max_observed * projection_scale
    return {
        "stream_capacity_eligible": configured + 1e-12 >= projected,
        "stream_capacity": {
            "trace_file": str(trace_path.resolve()),
            "trace_sha256": sha256_file(trace_path),
            "development_edit_count": dev_count,
            "formal_stream_edit_count": FORMAL_STREAM_EDIT_COUNT,
            "development_skipped_edits": skipped,
            "max_observed_cumulative_weight_drift_ratio": max_observed,
            "projection_scale": projection_scale,
            "projected_required_cumulative_weight_drift_ratio": projected,
            "configured_max_cumulative_weight_drift_ratio": configured,
            "projection_rule": "max_observed * sqrt(formal_stream_edits / development_edits)",
        },
    }


def check_disjoint(dev_path, test_path):
    dev, _ = load_manifest(dev_path)
    test, _ = load_manifest(test_path)
    for field in ("sample_id", "subject", "subject_id", "fact_fingerprint"):
        left = {_normalized_key(row[field]) for row in dev if row.get(field)}
        right = {_normalized_key(row[field]) for row in test if row.get(field)}
        overlap = left & right
        if overlap:
            raise ValueError(f"Dev/test overlap in {field}: {len(overlap)} normalized value(s); "
                             f"examples={sorted(overlap)[:3]!r}")
    fields = ("update", "rephrase", "personas", "loc", "mhop")
    left = {_normalized_key(row[field]) for row in dev for field in fields if row.get(field)}
    right = {_normalized_key(row[field]) for row in test for field in fields if row.get(field)}
    if left & right:
        raise ValueError("Dev/test overlap in evaluation questions")
    return len(dev)


def candidate_evidence(path, test_manifest, method):
    config_path, metric_path, completed_path = [path / name for name in
                                               ("run_config.json", "metrics.json", "completed.json")]
    config = json.loads(config_path.read_text())
    completed = json.loads(completed_path.read_text())
    family = _method_family(method)
    requested_method = _canonical_method(method).strip().upper().replace("_", "-")
    if (str(config.get("method", "")).strip().upper().replace("_", "-") != requested_method or
            completed.get("status") != "complete" or
            (completed.get("method") is not None and
             str(completed["method"]).strip().upper().replace("_", "-") != requested_method)):
        raise ValueError("Unfinished candidate or mismatched method")
    if family != "legacy":
        _require_development_markers(config, completed)
    layers = _layers(config)
    loss_layer = _loss_layer(config, method, layers)
    if family == "memit":
        provenance = config.get("statistics_provenance")
        if provenance is None:
            # Older MEMIT snapshots used projection_provenance.  Keep that
            # compatibility only for non-MEMIT callers; MEMIT must bind its
            # actual second-moment source files.
            raise ValueError("Candidate has no statistics provenance")
        statistics_provenance = _validate_file_provenance(
            provenance, "statistics provenance", layers
        )
    elif family == "rome" and _rome_mom2_adjustment(config) is True:
        audit = config.get("mom2_audit")
        if audit is None:
            raise ValueError("ROME mom2_adjustment=true requires mom2_audit")
        statistics_provenance = _validate_file_provenance(
            audit, "ROME mom2 audit", layers
        )
    elif family == "rome":
        statistics_provenance = None
    else:
        provenance = config.get("statistics_provenance", config.get("projection_provenance", []))
        _validate_file_provenance(provenance, "statistics provenance")
        statistics_provenance = None
    dev_manifest = Path(config["manifest"])
    count = check_disjoint(dev_manifest, test_manifest)
    if config["manifest_sha256"] != sha256_file(dev_manifest):
        raise ValueError("Changed development manifest")
    if config["num_facts"] != count:
        raise ValueError("Development candidate used a prefix only")
    if family == "legacy" and completed["num_facts"] != count:
        raise ValueError("Development candidate used a prefix only")
    if family != "legacy" and completed.get("num_facts", count) != count:
        raise ValueError("Development candidate used a prefix only")
    stage_counts = config.get("stage_counts")
    if stage_counts is not None:
        try:
            normalized_stage_counts = {int(stage): int(size) for stage, size in stage_counts.items()}
        except (AttributeError, TypeError, ValueError) as error:
            raise ValueError("Invalid candidate stage_counts") from error
        if (set(normalized_stage_counts) != set(range(1, 9)) or
                any(size <= 0 for size in normalized_stage_counts.values()) or
                sum(normalized_stage_counts.values()) != count):
            raise ValueError("Development candidate requires all eight complete stages")
    if completed.get("final_stage", 8) != 8:
        raise ValueError("Development candidate requires all eight stages")
    metrics = json.loads(metric_path.read_text())
    metrics = [record for record in metrics if record.get("current_stage") is not None]
    if [record["current_stage"] for record in metrics] != list(range(1, 9)):
        raise ValueError("Development candidate requires all eight stages")
    final = metrics[-1]
    if final["num_facts"] != count:
        raise ValueError("Final development panel incomplete")
    scores = [final[axis]["match_accuracy"] for axis in AXES]
    if not all(math.isfinite(score) and 0 <= score <= 1 for score in scores):
        raise ValueError("Invalid development scores")
    quality_trajectory = (
        _memit_quality_trajectory(metrics) if family == "memit" else {}
    )
    settings = _settings(config, method)
    # A ROME run stores most EasyEdit values under upstream_hparams.  Record
    # only stable algorithmic values in a separate certificate section so
    # every value in ``settings`` remains directly verifiable on runner args.
    hyperparameters = _rome_hyperparameters(config) if family == "rome" else {}
    if family == "rome":
        settings.setdefault("loss_layer", loss_layer)
        stream_capacity = _rome_stream_capacity(path, config, completed, count)
    else:
        stream_capacity = None
    result = {"run_dir": str(path.resolve()), "method": _canonical_method(method), "layers": layers,
            "dev_manifest": str(dev_manifest.resolve()), "dev_manifest_sha256": sha256_file(dev_manifest),
            "settings": settings,
            "model_dir": str(Path(config["model_dir"]).resolve()),
            "score": sum(scores) / len(scores), "axis_scores": dict(zip(AXES, scores)),
            "config_sha256": sha256_file(config_path), "metrics_sha256": sha256_file(metric_path),
            "completed_sha256": sha256_file(completed_path),
            **quality_trajectory}
    if stream_capacity is not None:
        result.update(stream_capacity)
    if family != "legacy":
        result["loss_layer"] = loss_layer
        result["hyperparameters"] = hyperparameters
        result["protocol_settings"] = _protocol_settings(config)
    if family != "rome":
        if "stats_dir" not in config:
            raise ValueError("Candidate has no stats_dir")
        result["stats_dir"] = str(Path(config["stats_dir"]).resolve())
    if statistics_provenance is not None:
        result["statistics_provenance"] = statistics_provenance
    return result


def select_candidates(paths, test_manifest, method):
    if len(paths) < 2:
        raise ValueError("A layer sweep needs at least two candidates")
    evidence = [candidate_evidence(Path(path), test_manifest, method) for path in paths]
    family = _method_family(method)
    common_fields = ("dev_manifest_sha256", "model_dir")
    if family != "legacy":
        # Protocol/evaluation invariants must be shared, while algorithmic
        # knobs (layers, loss layer, trust region, moments, ridges, etc.) are
        # intentionally allowed to vary across the sweep.
        common_fields += ("protocol_settings",)
    else:
        common_fields += ("settings",)
    if family == "legacy":
        common_fields += ("stats_dir",)
    for field in common_fields:
        if any(item[field] != evidence[0][field] for item in evidence):
            raise ValueError(f"Candidate {field} mismatch")
    if family == "legacy":
        identity = [(tuple(item["layers"]),) for item in evidence]
    else:
        identity = [
            (tuple(item["layers"]), _canonical_json(item.get("settings", {})),
             _canonical_json(item.get("hyperparameters", {})))
            for item in evidence
        ]
    if len(set(identity)) != len(identity):
        raise ValueError("Duplicate layer/configuration choices")
    if family == "memit":
        selectable = [item for item in evidence if item.get("quality_collapse") is True]
    else:
        selectable = [item for item in evidence if item.get("stream_capacity_eligible", True)]
    if not selectable:
        if family == "memit":
            raise ValueError(
                "No MEMIT candidate passes the quality trajectory gate: "
                f"drop <= {MEMIT_QUALITY_DROP_THRESHOLD} or "
                f"next/previous >= {MEMIT_QUALITY_RATIO_THRESHOLD}"
            )
        raise ValueError(
            "No ROME candidate has enough projected cumulative trust-region capacity "
            f"for the {FORMAL_STREAM_EDIT_COUNT}-edit stream"
        )
    best = min(
        selectable,
        key=lambda item: (
            -item["score"],
            tuple(item["layers"]),
            _canonical_json(item.get("settings", {})),
            _canonical_json(item.get("hyperparameters", {})),
        ),
    )
    certificate_format = ("editor-dev-layer-selection/v2" if family != "legacy"
                          else "editor-dev-layer-selection/v1")
    tie_break = ("canonical layers/settings/hyperparameters" if family != "legacy"
                 else "lexicographic layers")
    return {"format": certificate_format, "method": _canonical_method(method),
            "criterion": (
                f"final-dev-five-axis-match-macro-mean; ROME stream-capacity gate: "
                f"max_observed_cumulative_drift*sqrt({FORMAL_STREAM_EDIT_COUNT}/dev_edits) "
                f"<= configured max_cumulative_weight_drift_ratio; tie: {tie_break}"
                if family == "rome"
                else (
                    f"final-dev-five-axis-match-macro-mean; MEMIT quality-trajectory "
                    f"gate: reject if adjacent five-axis macro drop > "
                    f"{MEMIT_QUALITY_DROP_THRESHOLD} and next/previous < "
                    f"{MEMIT_QUALITY_RATIO_THRESHOLD}; tie: {tie_break}"
                    if family == "memit"
                    else f"final-dev-five-axis-match-macro-mean; tie: {tie_break}"
                )
            ),
            "test_manifest": str(test_manifest.resolve()), "test_manifest_sha256": sha256_file(test_manifest),
            "candidates": evidence, "selected": best,
            "independent_dev_verified": True, "test_scores_used": False}


def verify_selection(path, args, method, layers):
    if any(getattr(args, key, 0) for key in (
            "max_edits", "skip_eval", "max_stages", "examples_per_stage",
            "no_stage_eval", "locality_panel_size", "multihop_panel_size")):
        raise ValueError("Qualified runs require the complete manifest and full evaluation panels")
    if hasattr(args, "chunk_size"):
        rows, _ = load_manifest(args.manifest)
        from collections import Counter
        stage_sizes = Counter(row["stage"] for row in rows)
        if any(count != args.chunk_size for count in stage_sizes.values()) or args.eval_every_chunks != 1:
            raise ValueError("Qualified sequential CrispEdit requires one full time interval per chunk and evaluation at every interval")
    certificate = json.loads(path.read_text())
    supported_formats = {
        "editor-dev-layer-selection/v1",
        "editor-dev-layer-selection/v2",
        # Accepted for compatibility with an early development build of this
        # selector; newly generated certificates use the layer-selection name.
        "editor-dev-config-selection/v2",
    }
    if certificate.get("format") not in supported_formats:
        raise ValueError("Unsupported dev selection certificate")
    actual = select_candidates([item["run_dir"] for item in certificate["candidates"]], args.manifest, method)
    if actual != certificate:
        raise ValueError("Development selection evidence changed")
    selected = certificate["selected"]
    if list(layers) != selected["layers"]:
        raise ValueError("Requested layers do not match the dev selection")
    if str(Path(args.model_dir).resolve()) != selected["model_dir"]:
        raise ValueError("Selected model changed")
    if "stats_dir" in selected:
        if not hasattr(args, "stats_dir"):
            raise ValueError("Selected run requires stats_dir")
        if str(Path(args.stats_dir).resolve()) != selected["stats_dir"]:
            raise ValueError("Selected statistics directory changed")
    for key, value in selected["settings"].items():
        if not hasattr(args, key):
            raise ValueError(f"Requested run has no selected setting {key}")
        actual_value = _normalize_setting(key, getattr(args, key))
        if actual_value != value:
            raise ValueError(f"Requested {key} differs from the selected dev run")
    return {"path": str(path.resolve()), "sha256": sha256_file(path),
            "selected": selected, "criterion": certificate["criterion"],
            "independent_dev_verified": True, "test_scores_used": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = select_candidates(args.candidate_dirs, args.test_manifest, args.method)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["selected"], indent=2))


if __name__ == "__main__":
    main()
