from __future__ import annotations

import argparse
import copy
import csv
import itertools
import json
from pathlib import Path
import re
import shlex
from typing import Any

import yaml


def _load_structured(path: Path) -> dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping at top level of {path}.")
    return data


def _safe_slug(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    return value or "sweep"


def _short_value(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if value is None:
        return "none"
    if isinstance(value, (int, float, str)):
        return _safe_slug(str(value))
    if isinstance(value, list):
        return _safe_slug("-".join(str(item) for item in value))
    return _safe_slug(json.dumps(value, sort_keys=True))


def _get_path(data: dict[str, Any], dotted_path: str) -> Any:
    cur: Any = data
    for part in dotted_path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _set_path(data: dict[str, Any], dotted_path: str, value: Any) -> None:
    cur: dict[str, Any] = data
    parts = dotted_path.split(".")
    for part in parts[:-1]:
        next_value = cur.get(part)
        if next_value is None:
            next_value = {}
            cur[part] = next_value
        if not isinstance(next_value, dict):
            raise ValueError(f"Cannot set {dotted_path}: {part} already contains a non-mapping value.")
        cur = next_value
    cur[parts[-1]] = value


def _format_template(template: str, *, name: str, index: int, output_dir: Path, config_path: Path) -> str:
    return template.format(
        name=name,
        index=index,
        output_dir=str(output_dir).replace("\\", "/"),
        config_path=str(config_path).replace("\\", "/"),
    )


def _apply_templates(cfg: dict[str, Any], templates: dict[str, Any], *, name: str, index: int, output_dir: Path, config_path: Path) -> None:
    for dotted_path, raw_value in templates.items():
        if isinstance(raw_value, str):
            value = _format_template(raw_value, name=name, index=index, output_dir=output_dir, config_path=config_path)
        else:
            value = raw_value
        _set_path(cfg, dotted_path, value)


def _with_default_run_templates(templates: dict[str, Any], base_cfg: dict[str, Any]) -> dict[str, Any]:
    templates = dict(templates)
    if isinstance(base_cfg.get("train"), dict):
        templates.setdefault("train.log_file", "{output_dir}/runs/{name}/train.log")
    return templates


def _grid_items(spec: dict[str, Any]) -> list[tuple[str, list[Any]]]:
    grid = spec.get("grid")
    if not isinstance(grid, dict) or not grid:
        raise ValueError("Sweep spec must contain a non-empty 'grid' mapping.")
    items: list[tuple[str, list[Any]]] = []
    for key, values in grid.items():
        if not isinstance(values, list):
            raise ValueError(f"Grid value for '{key}' must be a list.")
        if not values:
            raise ValueError(f"Grid value for '{key}' is empty.")
        items.append((str(key), values))
    return items


def _name_for_combo(*, index: int, keys: list[str], values: tuple[Any, ...], name_fields: list[str] | None) -> str:
    value_by_key = dict(zip(keys, values))
    fields = name_fields if name_fields else keys
    parts = [f"{index:04d}"]
    for key in fields:
        if key not in value_by_key:
            continue
        label = key.split(".")[-1]
        parts.append(f"{_safe_slug(label)}-{_short_value(value_by_key[key])}")
    return "_".join(parts)


def _quote_ps(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _write_launchers(
    *,
    output_dir: Path,
    command: str,
    config_paths: list[Path],
    names: list[str],
) -> None:
    sh_lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    ps_lines = ["$ErrorActionPreference = 'Stop'", ""]
    for name, config_path in zip(names, config_paths):
        cfg_text = str(config_path).replace("\\", "/")
        sh_lines.append(f"echo '=== {name} ==='")
        sh_lines.append(command.format(config=shlex.quote(cfg_text), config_path=shlex.quote(cfg_text), name=shlex.quote(name)))
        sh_lines.append("")
        ps_lines.append(f"Write-Host {_quote_ps(f'=== {name} ===')}")
        ps_lines.append(command.format(config=_quote_ps(cfg_text), config_path=_quote_ps(cfg_text), name=_quote_ps(name)))
        ps_lines.append("")
    sh_path = output_dir / "run_all.sh"
    ps_path = output_dir / "run_all.ps1"
    sh_path.write_text("\n".join(sh_lines), encoding="utf-8")
    ps_path.write_text("\n".join(ps_lines), encoding="utf-8")


def _write_manifest(
    *,
    output_dir: Path,
    rows: list[dict[str, Any]],
    grid_keys: list[str],
) -> None:
    json_path = output_dir / "manifest.json"
    csv_path = output_dir / "manifest.csv"
    json_path.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
    fieldnames = ["index", "name", "config_path", "log_file"] + grid_keys
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(row[key], sort_keys=True) if key in grid_keys else row.get(key)
                    for key in fieldnames
                }
            )


def generate_sweep(args: argparse.Namespace) -> None:
    base_cfg = _load_structured(args.base_config)
    spec = _load_structured(args.spec)
    grid = _grid_items(spec)
    keys = [key for key, _ in grid]
    value_lists = [values for _, values in grid]
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    configs_dir = output_dir / str(spec.get("configs_dir", "configs"))
    configs_dir.mkdir(parents=True, exist_ok=True)

    command = str(args.command or spec.get("command") or "python {config}")
    name_fields_raw = spec.get("name_fields")
    name_fields = [str(item) for item in name_fields_raw] if isinstance(name_fields_raw, list) else None
    templates = _with_default_run_templates(dict(spec.get("templates") or {}), base_cfg)
    static_overrides = dict(spec.get("overrides") or {})

    rows: list[dict[str, Any]] = []
    config_paths: list[Path] = []
    names: list[str] = []
    for index, values in enumerate(itertools.product(*value_lists)):
        cfg = copy.deepcopy(base_cfg)
        for key, value in static_overrides.items():
            _set_path(cfg, str(key), value)
        for key, value in zip(keys, values):
            _set_path(cfg, key, value)
        name = _name_for_combo(index=index, keys=keys, values=values, name_fields=name_fields)
        config_path = configs_dir / f"{name}.yaml"
        _apply_templates(cfg, templates, name=name, index=index, output_dir=output_dir, config_path=config_path)
        config_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=False), encoding="utf-8")
        row = {
            "index": index,
            "name": name,
            "config_path": str(config_path).replace("\\", "/"),
        }
        log_file = _get_path(cfg, "train.log_file")
        if log_file:
            row["log_file"] = str(log_file).replace("\\", "/")
        row.update({key: value for key, value in zip(keys, values)})
        rows.append(row)
        config_paths.append(config_path)
        names.append(name)

    _write_manifest(output_dir=output_dir, rows=rows, grid_keys=keys)
    _write_launchers(output_dir=output_dir, command=command, config_paths=config_paths, names=names)
    print(f"Wrote {len(rows)} configs to {configs_dir}")
    print(f"Wrote {output_dir / 'manifest.csv'}")
    print(f"Wrote {output_dir / 'run_all.ps1'}")
    print(f"Wrote {output_dir / 'run_all.sh'}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate expanded YAML configs, manifests, and launchers from a grid sweep spec."
    )
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True, help="YAML/JSON file with grid, templates, and command.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--command",
        type=str,
        default=None,
        help="Launcher command template. Use {config} or {config_path}; overrides spec.command.",
    )
    args = parser.parse_args()
    generate_sweep(args)


if __name__ == "__main__":
    main()
