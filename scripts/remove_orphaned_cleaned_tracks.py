"""Remove generated cleaned tracks whose folder has no tracks.txt source.

Usage:
    python scripts/remove_orphaned_cleaned_tracks.py --fred-root D:/datasets/FRED
    python scripts/remove_orphaned_cleaned_tracks.py --fred-root D:/datasets/FRED --dry-run

Only immediate subdirectories of the dataset root are checked. No backups
are created.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def remove_orphaned_cleaned_tracks(root: Path, *, dry_run: bool = False) -> int:
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(f"Dataset root is not a directory: {root}")

    removed = 0
    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue
        # Avoid traversing directory links or junctions outside the dataset.
        try:
            folder.resolve(strict=True).relative_to(root)
        except ValueError:
            print(f"Skipping folder outside dataset root: {folder}")
            continue
        cleaned = folder / "cleaned_tracks.txt"
        original = folder / "tracks.txt"
        if not cleaned.is_file() or original.is_file():
            continue
        if dry_run:
            print(f"Would remove: {cleaned}")
        else:
            cleaned.unlink()
            print(f"Removed: {cleaned}")
        removed += 1
    return removed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--fred-root", type=Path, default=Path("datasets/FRED"),
        help="Dataset root (default: datasets/FRED).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="List files that would be removed without deleting them.",
    )
    args = parser.parse_args()
    try:
        count = remove_orphaned_cleaned_tracks(args.fred_root, dry_run=args.dry_run)
    except OSError as exc:
        parser.exit(1, f"Error: {exc}\n")
    action = "Would remove" if args.dry_run else "Removed"
    print(f"{action} {count} orphaned cleaned_tracks.txt file(s).")


if __name__ == "__main__":
    main()
