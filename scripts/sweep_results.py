from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable

import yaml


@dataclass
class SweepRun:
    index: int
    name: str
    config_path: Path
    manifest: dict[str, Any]
    config: dict[str, Any]
    test_results: dict[str, Any] | None
    metrics_rows: list[dict[str, Any]]
    flat: dict[str, Any]


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping in {path}.")
    return data


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}.")
    return data


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        item = json.loads(line)
        if isinstance(item, dict):
            rows.append(item)
    return rows


def flatten_mapping(data: Any, *, prefix: str = "", sep: str = ".") -> dict[str, Any]:
    if isinstance(data, dict):
        out: dict[str, Any] = {}
        for key, value in data.items():
            child_prefix = f"{prefix}{sep}{key}" if prefix else str(key)
            out.update(flatten_mapping(value, prefix=child_prefix, sep=sep))
        return out
    return {prefix: data}


def _resolve_path(path_value: Any, *, base_dir: Path) -> Path | None:
    if path_value is None:
        return None
    path = Path(str(path_value))
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _metric_improved(value: float, best: float | None, *, mode: str) -> bool:
    if best is None:
        return True
    if mode == "min":
        return value < best
    if mode == "max":
        return value > best
    raise ValueError(f"Unknown metric mode: {mode}")


def _metrics_summary(rows: list[dict[str, Any]], cfg: dict[str, Any]) -> dict[str, Any]:
    if not rows:
        return {}
    train_cfg = dict(cfg.get("train") or {})
    split = str(train_cfg.get("best_metric_split", "val"))
    metric = str(train_cfg.get("best_metric", "loss"))
    mode = str(train_cfg.get("best_metric_mode", "min"))
    best_value: float | None = None
    best_row: dict[str, Any] | None = None
    for row in rows:
        split_metrics = row.get(split)
        if not isinstance(split_metrics, dict) or metric not in split_metrics:
            continue
        value = float(split_metrics[metric])
        if _metric_improved(value, best_value, mode=mode):
            best_value = value
            best_row = row
    out: dict[str, Any] = {
        "metrics.num_epochs_logged": len(rows),
        "metrics.last_epoch": rows[-1].get("epoch"),
    }
    out.update({f"metrics.last.{key}": value for key, value in flatten_mapping(rows[-1]).items()})
    if best_row is not None:
        out["metrics.best_epoch"] = best_row.get("epoch")
        out["metrics.best_metric_split"] = split
        out["metrics.best_metric"] = metric
        out["metrics.best_metric_mode"] = mode
        out["metrics.best_metric_value"] = best_value
        out.update({f"metrics.best.{key}": value for key, value in flatten_mapping(best_row).items()})
    return out


def _read_manifest(manifest_path: Path) -> list[dict[str, Any]]:
    if manifest_path.suffix.lower() == ".json":
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"Expected list in {manifest_path}.")
        return [dict(row) for row in data]
    rows: list[dict[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = {}
            for key, value in row.items():
                if value is None:
                    parsed[key] = value
                    continue
                try:
                    parsed[key] = json.loads(value)
                except json.JSONDecodeError:
                    parsed[key] = value
            rows.append(parsed)
    return rows


def load_sweep_runs(*, sweep_dir: Path | None = None, manifest_path: Path | None = None) -> list[SweepRun]:
    if manifest_path is None:
        if sweep_dir is None:
            raise ValueError("Either sweep_dir or manifest_path is required.")
        manifest_path = Path(sweep_dir) / "manifest.json"
        if not manifest_path.exists():
            manifest_path = Path(sweep_dir) / "manifest.csv"
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Sweep manifest not found: {manifest_path}")
    base_dir = manifest_path.parent.resolve()
    runs: list[SweepRun] = []
    for row in _read_manifest(manifest_path):
        config_path = _resolve_path(row.get("config_path"), base_dir=Path.cwd())
        if config_path is None:
            raise ValueError(f"Manifest row is missing config_path: {row}")
        cfg = _load_yaml(config_path)
        train_cfg = dict(cfg.get("train") or {})
        test_path = _resolve_path(train_cfg.get("test_metrics_json"), base_dir=Path.cwd())
        metrics_path = _resolve_path(train_cfg.get("metrics_jsonl"), base_dir=Path.cwd())
        test_results = _load_json(test_path) if test_path is not None and test_path.exists() else None
        metrics_rows = _load_jsonl(metrics_path) if metrics_path is not None else []
        flat: dict[str, Any] = {
            "index": int(row.get("index", len(runs))),
            "name": str(row.get("name", config_path.stem)),
            "config_path": str(config_path).replace("\\", "/"),
            "test_metrics_json": None if test_path is None else str(test_path).replace("\\", "/"),
            "metrics_jsonl": None if metrics_path is None else str(metrics_path).replace("\\", "/"),
            "test_results_found": bool(test_results is not None),
            "metrics_jsonl_found": bool(metrics_rows),
        }
        flat.update({f"manifest.{key}": value for key, value in row.items()})
        flat.update({f"config.{key}": value for key, value in flatten_mapping(cfg).items()})
        if test_results is not None:
            flat.update({f"test_json.{key}": value for key, value in flatten_mapping(test_results).items()})
            test_metrics = test_results.get("test")
            if isinstance(test_metrics, dict):
                flat.update({f"test.{key}": value for key, value in flatten_mapping(test_metrics).items()})
        flat.update(_metrics_summary(metrics_rows, cfg))
        runs.append(
            SweepRun(
                index=int(flat["index"]),
                name=str(flat["name"]),
                config_path=config_path,
                manifest=row,
                config=cfg,
                test_results=test_results,
                metrics_rows=metrics_rows,
                flat=flat,
            )
        )
    return runs


def sweep_records(*, sweep_dir: Path | None = None, manifest_path: Path | None = None) -> list[dict[str, Any]]:
    return [run.flat for run in load_sweep_runs(sweep_dir=sweep_dir, manifest_path=manifest_path)]


def sweep_dataframe(*, sweep_dir: Path | None = None, manifest_path: Path | None = None):
    import pandas as pd

    return pd.DataFrame(sweep_records(sweep_dir=sweep_dir, manifest_path=manifest_path))


def _stringify_table_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return value


def write_records_csv(records: Iterable[dict[str, Any]], path: Path) -> None:
    records = list(records)
    keys = sorted({key for record in records for key in record})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for record in records:
            writer.writerow({key: _stringify_table_value(record.get(key)) for key in keys})


def write_records_json(records: Iterable[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(list(records), indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read generated sweep configs/results into table-ready records."
    )
    parser.add_argument("--sweep-dir", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--print-columns", action="store_true")
    parser.add_argument("--require-test", action="store_true", help="Only include runs with test_metrics_json present.")
    args = parser.parse_args()
    records = sweep_records(sweep_dir=args.sweep_dir, manifest_path=args.manifest)
    if args.require_test:
        records = [record for record in records if bool(record.get("test_results_found"))]
    if args.output_csv is not None:
        write_records_csv(records, args.output_csv)
        print(f"Wrote {args.output_csv}")
    if args.output_json is not None:
        write_records_json(records, args.output_json)
        print(f"Wrote {args.output_json}")
    if args.print_columns:
        for key in sorted({key for record in records for key in record}):
            print(key)
    print(f"Loaded {len(records)} sweep runs.")


if __name__ == "__main__":
    main()
