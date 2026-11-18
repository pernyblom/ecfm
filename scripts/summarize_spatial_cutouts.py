"""Build comparison.png from the spatial cutout inspector's summary.json files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]


def _image_path(summary_path: Path, recorded_path: str) -> Path:
    path = Path(recorded_path.replace('\\', '/'))
    # The inspector records paths relative to its working directory. Also
    # support moving a complete export directory, keeping its sample folders.
    candidates = [summary_path.parent / path.parent.name / path.name,
                  summary_path.parent / path, path, ROOT / path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f'Cutout from {summary_path} not found: {recorded_path}')


def create_comparison(input_dir: Path, *, output: Path | None = None,
                      recursive: bool = False, samples_per_summary: int | None = None,
                      representations: list[str] | None = None,
                      tile_size: tuple[int, int] = (256, 256)) -> Path:
    input_dir = Path(input_dir)
    if min(tile_size) < 1:
        raise ValueError('Tile dimensions must be positive')
    if samples_per_summary is not None and samples_per_summary < 1:
        raise ValueError('samples_per_summary must be positive')
    summaries = sorted(input_dir.rglob('summary.json')) if recursive else [input_dir / 'summary.json']
    rows = []
    available_reps = []
    for summary_path in summaries:
        summary = json.loads(summary_path.read_text(encoding='utf-8'))
        for rep in summary.get('representations', []):
            if rep not in available_reps:
                available_reps.append(rep)
        for sample in summary['samples'][:samples_per_summary]:
            rows.append((summary_path, sample))
            for rep in sample['outputs']:
                if rep not in available_reps:
                    available_reps.append(rep)
    if not rows:
        raise ValueError(f'No cutout samples found in {input_dir}')
    reps = representations if representations is not None else available_reps
    if not reps or set(reps) - set(available_reps):
        raise ValueError(f'Choose representations from {available_reps}')

    padding, header_height, label_height = 8, 28, 28
    # Keep native pixels for small cutouts and preserve aspect ratios for larger
    # previews, including channel grids. Never renormalize the saved images.
    column_widths = [tile_size[0]] * len(reps)
    measure = ImageDraw.Draw(Image.new('RGB', (1, 1)))
    for col, rep in enumerate(reps):
        bounds = measure.textbbox((0, 0), rep)
        column_widths[col] = max(column_widths[col], bounds[2] - bounds[0])
    row_height = tile_size[1] + label_height + padding
    canvas = Image.new('RGB', (sum(column_widths) + padding*(len(reps)+1),
                              header_height + row_height*len(rows)), (22, 22, 22))
    draw = ImageDraw.Draw(canvas)
    x = padding
    for rep, width in zip(reps, column_widths):
        draw.text((x, padding), rep, fill='white')
        x += width + padding
    for row, (summary_path, sample) in enumerate(rows):
        top = header_height + row*row_height
        label = f"FRED {sample['frame_key']} | track {sample['track_id']}"
        if 'anchor_time_s' in sample:
            label += f" | t={sample['anchor_time_s']:.6f} s"
        draw.text((padding, top+4), label, fill='white')
        x = padding
        for rep, width in zip(reps, column_widths):
            entry = sample['outputs'].get(rep)
            if entry is None:
                draw.text((x, top+label_height), 'Not available', fill=(150, 150, 150))
            else:
                path = _image_path(summary_path, entry['cutout'])
                with Image.open(path) as source:
                    tile = source.convert('RGB')
                tile.thumbnail(tile_size, Image.Resampling.LANCZOS)
                canvas.paste(tile, (x, top+label_height))
            x += width + padding
    output = Path(output) if output is not None else input_dir / 'comparison.png'
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=Path('outputs/fred_projected_stacks'))
    parser.add_argument('--output', type=Path, help='Default: <input-dir>/comparison.png')
    parser.add_argument('--recursive', action='store_true', help='Include summary.json files in subdirectories')
    parser.add_argument('--samples-per-summary', type=int, help='Use only the first N samples from each summary')
    parser.add_argument('--representations', nargs='+', help='Columns in this order; default: all representations')
    parser.add_argument('--tile-size', type=int, nargs=2, default=(256, 256), metavar=('WIDTH', 'HEIGHT'),
                        help='Maximum preview size, preserving aspect ratio (default: 256 256)')
    args = parser.parse_args()
    try:
        output = create_comparison(args.input_dir, output=args.output, recursive=args.recursive,
            samples_per_summary=args.samples_per_summary, representations=args.representations,
            tile_size=tuple(args.tile_size))
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(1, f'Error: {exc}\n')
    print(f'Wrote comparison: {output}')


if __name__ == '__main__':
    main()
