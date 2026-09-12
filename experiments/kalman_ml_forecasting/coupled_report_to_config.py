"""Convert an optimized coupled-Kalman JSON report to YAML configuration."""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any

import yaml


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
        model_cfg["rollout_transition"] = "configured_kalman"

    output["kalman"]["dynamics_generator"] = [
        _FlowList(row) for row in output["kalman"]["dynamics_generator"]
    ]
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
