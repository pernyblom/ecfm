from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path


def _read_timestamps(path: Path) -> Counter:
    counts: Counter = Counter()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        ts = line.split(":", 1)[0].strip()
        if not ts:
            continue
        counts[ts] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check coordinates.txt for duplicate timestamps per folder."
    )
    parser.add_argument(
        "--fred-root",
        type=Path,
        default=Path("datasets/FRED"),
        help="Root path of FRED dataset (default: datasets/FRED)",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="coordinates.txt",
        help="Filename to check (default: coordinates.txt)",
    )
    args = parser.parse_args()

    root = args.fred_root
    folders = sorted([p for p in root.iterdir() if p.is_dir() and p.name.isdigit()], key=lambda p: int(p.name))
    any_dupes = False
    for folder in folders:
        path = folder / args.filename
        if not path.exists():
            continue
        counts = _read_timestamps(path)
        dupes = {ts: c for ts, c in counts.items() if c > 1}
        if dupes:
            any_dupes = True
            total = sum(dupes.values())
            uniq = len(dupes)
            worst = max(dupes.values())
            print(f"{folder.name}: {uniq} duplicated timestamps, {total} entries, max dup {worst}")
    if not any_dupes:
        print("No duplicate timestamps found.")


if __name__ == "__main__":
    main()
