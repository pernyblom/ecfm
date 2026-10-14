"""Convert an optimized Kalman JSON report to a runtime/residual YAML config."""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.kalman_ml_forecasting.models.kalman_filter import kalman_config_from_dict

NOISE_KEYS = (
    "initial_pos_std",
    "initial_size_std",
    "initial_vel_std",
    "process_pos_std",
    "process_size_std",
    "process_vel_std",
    "process_size_vel_std",
    "measurement_pos_std",
    "measurement_size_std",
)
EXPECTED_LAYOUT = ["cx", "cy", "w", "h", "vx", "vy", "vw", "vh"]


class _FlowList(list):
    pass


class _ConfigDumper(yaml.SafeDumper):
    pass


def _represent_flow_list(dumper: yaml.SafeDumper, value: _FlowList):
    return dumper.represent_sequence("tag:yaml.org,2002:seq", value, flow_style=True)


_ConfigDumper.add_representer(_FlowList, _represent_flow_list)


def kalman_config_from_report(report: dict[str, Any]) -> dict[str, Any]:
    # Scalar optimizers (including the existing 12-D CA optimizer) export best.params.
    best = report.get("best")
    if isinstance(best, dict) and "model" not in best and isinstance(best.get("params"), dict):
        return kalman_config_from_dict(best["params"])
    try:
        model = report["best"]["model"]
    except (KeyError, TypeError) as exc:
        raise ValueError("Report must contain best.model.") from exc
    if not isinstance(model, dict):
        raise ValueError("Report best.model must be a mapping.")

    layout = model.get("state_layout")
    if layout != EXPECTED_LAYOUT:
        raise ValueError(
            f"Expected coupled state_layout {EXPECTED_LAYOUT}, got {layout!r}."
        )
    if model.get("schema_version") == 2:
        config = model.get("kalman_config")
        if not isinstance(config, dict):
            raise ValueError("Version 2 report requires best.model.kalman_config.")
        if config.get("motion_model") not in {"constant_velocity", "coupled"}:
            raise ValueError("Version 2 variant reports require an 8-D motion model.")
        return kalman_config_from_dict(config)
    if model.get("schema_version") not in (None, 1):
        raise ValueError(f"Unsupported Kalman report schema_version: {model.get('schema_version')}.")
    parameterization = str(model.get("parameterization", ""))
    if "matrix_exp" not in parameterization or "dynamics_generator" not in parameterization:
        raise ValueError(f"Unsupported coupled parameterization: {parameterization!r}.")

    generator = model.get("dynamics_generator")
    if not isinstance(generator, list) or len(generator) != 8:
        raise ValueError("best.model.dynamics_generator must contain eight rows.")
    converted_generator: list[list[float]] = []
    for row in generator:
        if not isinstance(row, list) or len(row) != 8:
            raise ValueError("best.model.dynamics_generator must be an 8x8 matrix.")
        converted_row = [float(value) for value in row]
        if not all(math.isfinite(value) for value in converted_row):
            raise ValueError("best.model.dynamics_generator contains a non-finite value.")
        converted_generator.append(converted_row)

    noise = model.get("noise")
    if not isinstance(noise, dict):
        raise ValueError("Report best.model.noise must be a mapping.")
    missing = [key for key in NOISE_KEYS if key not in noise]
    if missing:
        raise ValueError(f"Report best.model.noise is missing keys: {missing}.")
    converted_noise = {key: float(noise[key]) for key in NOISE_KEYS}
    if not all(math.isfinite(value) and value > 0 for value in converted_noise.values()):
        raise ValueError("All reported Kalman noise standard deviations must be finite and positive.")

    return {
        "enabled": True,
        "motion_model": "coupled",
        "dynamics_generator": converted_generator,
        **converted_noise,
    }


def _load_mapping(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    value = json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a mapping at the top level of {path}.")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="JSON report from optimize_coupled_kalman.py")
    parser.add_argument(
        "--base-config",
        type=Path,
        help="Optional YAML config whose kalman section will be replaced.",
    )
    parser.add_argument("--output", type=Path, help="Write YAML here instead of stdout.")
    parser.add_argument("--training-output-dir", type=Path,
                        help="Set separate residual checkpoint/log/metrics paths and clear resume_checkpoint.")
    parser.add_argument(
        "--use-for-residual-rollout",
        action="store_true",
        help=(
            "Also set model.initial_state_source=kalman_filter and "
            "model.rollout_transition=configured_kalman."
        ),
    )
    parser.add_argument("--force", action="store_true", help="Allow overwriting --output.")
    args = parser.parse_args()

    report = _load_mapping(args.report)
    kalman_cfg = kalman_config_from_report(report)
    if args.base_config is None:
        output: dict[str, Any] = {"kalman": kalman_cfg}
    else:
        output = copy.deepcopy(_load_mapping(args.base_config))
        output["kalman"] = kalman_cfg
    if args.use_for_residual_rollout:
        model_cfg = output.setdefault("model", {})
        if not isinstance(model_cfg, dict):
            raise ValueError("The base config's model section must be a mapping.")
        model_cfg["initial_state_source"] = "kalman_filter"
        # CA remains available for history/baseline; residual rollout is still 8-D.
        model_cfg["rollout_transition"] = (
            "constant_velocity" if kalman_cfg["motion_model"] == "constant_acceleration" else "configured_kalman"
        )
    if args.training_output_dir is not None:
        train_cfg = output.setdefault("train", {})
        if not isinstance(train_cfg, dict):
            raise ValueError("The base config's train section must be a mapping.")
        directory = args.training_output_dir
        train_cfg.update(checkpoint_dir=str(directory), metrics_jsonl=str(directory / "metrics.jsonl"),
                         log_file=str(directory / "train.log"), test_metrics_json=str(directory / "best_test_metrics.json"),
                         resume_checkpoint=None)

    for key in ("dynamics_generator", "initial_covariance", "process_covariance", "measurement_covariance"):
        if output["kalman"].get(key) is not None:
            output["kalman"][key] = [_FlowList(row) for row in output["kalman"][key]]
    rendered = yaml.dump(
        output, Dumper=_ConfigDumper, sort_keys=False, allow_unicode=False, width=240
    )
    if args.output is None:
        print(rendered, end="")
        return
    if args.output.exists() and not args.force:
        raise FileExistsError(f"Refusing to overwrite {args.output}; pass --force to replace it.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
