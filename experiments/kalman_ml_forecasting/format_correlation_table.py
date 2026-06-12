from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_SPLIT_ORDER = ["train", "train_eval", "test"]
DEFAULT_DECORR_ORDER = ["no", "yes"]
SPLIT_LABELS = {
    "train": "Train",
    "train_eval": "Train-eval",
    "val": "Val",
    "test": "Test",
}


def _load_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        ]
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as f:
            return [dict(row) for row in csv.DictReader(f)]
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(data, dict) and isinstance(data.get("rows"), list):
        return list(data["rows"])
    if isinstance(data, list):
        return list(data)
    raise ValueError(f"Could not find table rows in {path}.")


def _as_float(row: dict[str, Any], key: str) -> float:
    value = row.get(key)
    if value is None or value == "":
        raise ValueError(f"Missing required value '{key}' in row: {row}")
    return float(value)


def _as_int(row: dict[str, Any], key: str) -> int:
    value = row.get(key)
    if value is None or value == "":
        raise ValueError(f"Missing required value '{key}' in row: {row}")
    return int(float(value))


def _split_label(split: str) -> str:
    return SPLIT_LABELS.get(split, split.replace("_", r"\_"))


def _format_float(value: float, digits: int) -> str:
    return f"{value:.{digits}f}"


def _format_samples(value: int) -> str:
    return f"{value:,}"


def _ordered_rows(rows: list[dict[str, Any]], split_order: list[str]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        split = str(row.get("split", ""))
        decorrelated = str(row.get("decorrelated", "")).lower()
        if decorrelated in {"true", "1"}:
            decorrelated = "yes"
        elif decorrelated in {"false", "0"}:
            decorrelated = "no"
        by_key[(split, decorrelated)] = row

    out = []
    seen: set[tuple[str, str]] = set()
    for split in split_order:
        for decorrelated in DEFAULT_DECORR_ORDER:
            key = (split, decorrelated)
            if key in by_key:
                out.append(by_key[key])
                seen.add(key)
    remaining = [
        row
        for key, row in sorted(by_key.items(), key=lambda item: item[0])
        if key not in seen
    ]
    return out + remaining


def _latex_row(row: dict[str, Any], *, metric_digits: int, accel_digits: int) -> str:
    split = _split_label(str(row.get("split", "")))
    decorrelated = str(row.get("decorrelated", "")).lower()
    if decorrelated in {"true", "1"}:
        decorrelated = "yes"
    elif decorrelated in {"false", "0"}:
        decorrelated = "no"
    samples = _format_samples(_as_int(row, "samples"))
    mean_abs_corr = _format_float(_as_float(row, "mean_abs_corr"), metric_digits)
    mean_r2 = _format_float(_as_float(row, "mean_r2"), metric_digits)
    mean_accel_norm = _format_float(_as_float(row, "mean_accel_norm"), accel_digits)
    return (
        f"    {split} & {decorrelated} & {samples} & {mean_abs_corr} & "
        f"{mean_r2} & {mean_accel_norm} \\\\"
    )


def format_table(
    rows: list[dict[str, Any]],
    *,
    split_order: list[str],
    metric_digits: int,
    accel_digits: int,
    caption: str,
    label: str,
    table_env: bool,
) -> str:
    body_rows = [
        _latex_row(row, metric_digits=metric_digits, accel_digits=accel_digits)
        for row in _ordered_rows(rows, split_order)
    ]
    lines = []
    if table_env:
        lines.extend(
            [
                r"\begin{table}[t]",
                r"  \centering",
                f"  \\caption{{{caption}}}",
                f"  \\label{{{label}}}",
            ]
        )
    lines.extend(
        [
            r"  \begin{tabular}{lccccc}",
            r"    \toprule",
            r"    Split & Decorr. & Samples & Mean $|\rho|$ $\downarrow$ & Mean $R^2$ $\downarrow$ & $\|\bar{\mathbf{a}}\|_2$ $\downarrow$ \\",
            r"    \midrule",
            *body_rows,
            r"    \bottomrule",
            r"  \end{tabular}",
        ]
    )
    if table_env:
        lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Format correlation_table_experiment output as a LaTeX table."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to correlation_table_summary.json, correlation_table_rows.jsonl, or correlation_table_rows.csv.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .tex path. If omitted, writes to stdout.",
    )
    parser.add_argument(
        "--split-order",
        type=str,
        default=",".join(DEFAULT_SPLIT_ORDER),
        help="Comma-separated split order for the table body.",
    )
    parser.add_argument("--metric-digits", type=int, default=3)
    parser.add_argument("--accel-digits", type=int, default=3)
    parser.add_argument(
        "--caption",
        type=str,
        default="Motion-prior predictability of fitted future acceleration before and after sample decorrelation.",
    )
    parser.add_argument("--label", type=str, default="tab:correlation")
    parser.add_argument(
        "--fragment",
        action="store_true",
        help="Write only the tabular environment, without table/caption/label.",
    )
    args = parser.parse_args()

    rows = _load_rows(args.input)
    table = format_table(
        rows,
        split_order=[part.strip() for part in args.split_order.split(",") if part.strip()],
        metric_digits=max(0, int(args.metric_digits)),
        accel_digits=max(0, int(args.accel_digits)),
        caption=args.caption,
        label=args.label,
        table_env=not bool(args.fragment),
    )
    if args.output is None:
        print(table, end="")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(table, encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
