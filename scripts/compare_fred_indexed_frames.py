"""Sample labeled UAV frames, regenerate via indexed RAW, and build a visual report."""
import argparse
import html
import json
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.render_evt3_yolo_frames import ensure_rendered_frames, _read_yolo_boxes


def panel(image, title, size=(480, 300)):
    canvas = Image.new('RGB', size, '#171c25')
    image = image.copy()
    image.thumbnail((size[0], size[1]-24))
    canvas.paste(image, ((size[0]-image.width)//2, 24+(size[1]-24-image.height)//2))
    ImageDraw.Draw(canvas).text((8, 6), title, fill='white')
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-root', type=Path, default=Path('outputs/fred_reps'))
    parser.add_argument('--output-root', type=Path, default=Path('outputs/fred_reps_indexed_33ms'))
    parser.add_argument('--fred-root', type=Path, default=Path('datasets/FRED'))
    parser.add_argument('--sequences', type=int, default=4)
    parser.add_argument('--frames-per-sequence', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.output_root.resolve() == args.reference_root.resolve():
        raise ValueError('Comparison output must differ from reference images')
    if args.sequences < 1 or args.frames_per_sequence < 1:
        raise ValueError('Sample counts must be positive')
    rng = np.random.default_rng(args.seed)
    folders = sorted(p for p in args.reference_root.iterdir() if p.is_dir() and p.name.isdigit())
    rng.shuffle(folders)
    selected = []
    reps = ['cstr3', 'xt_my', 'yt_mx']
    for folder in folders:
        raw = args.fred_root / folder.name / 'Event/events.raw'
        manifest_path = folder / 'render_manifest.json'
        if not raw.exists() or not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text())
        params = manifest['render_params']
        if params['window'] != 33333 or params['window_mode'] != 'trailing':
            continue
        if not all(r in params['representation'] for r in reps):
            continue
        entries = list(manifest['files'])
        rng.shuffle(entries)
        usable = []
        for entry in entries:
            label = args.fred_root / folder.name / 'Event_YOLO' / (entry['label_stem']+'.txt')
            if not entry.get('num_boxes') or not label.is_file():
                continue
            if not all(r in entry['representations'] for r in reps):
                continue
            if not all((folder / f"{entry['label_stem']}_{r}.png").is_file() for r in reps):
                continue
            boxes = _read_yolo_boxes(label)
            # Favor UAVs big enough to inspect, without choosing particular times.
            if not boxes or max(w*h for _, _, w, h in boxes) < 0.0003:
                continue
            usable.append((entry, boxes))
            if len(usable) == args.frames_per_sequence:
                break
        if len(usable) == args.frames_per_sequence:
            selected.append((folder, manifest, usable))
        if len(selected) == args.sequences:
            break
    if len(selected) != args.sequences:
        raise ValueError(f'Only {len(selected)} eligible sequences found')
    args.output_root.mkdir(parents=True, exist_ok=True)
    preview_dir = args.output_root / 'comparisons'
    preview_dir.mkdir(exist_ok=True)
    report = dict(seed=args.seed, reference_root=str(args.reference_root), frames=[])
    links = []
    for folder, manifest, usable in selected:
        params = dict(manifest['render_params'])
        params['representation'] = reps
        params['image_sizes'] = {r: size for r, size in params.get('image_sizes', {}).items() if r in reps}
        params['crop_representations'] = [r for r in params.get('crop_representations', []) if r in reps]
        new_manifest = ensure_rendered_frames(
            args.fred_root / folder.name / 'Event/events.raw',
            args.fred_root / folder.name / 'Event_YOLO', args.output_root / folder.name,
            [e['label_stem'] for e, _ in usable], params)
        new_entries = {e['label_stem']: e for e in new_manifest['files']}
        for entry, boxes in usable:
            stem = entry['label_stem']
            new_entry = new_entries[stem]
            result = dict(sequence=folder.name, stem=stem, boxes=boxes,
                          old_event_count=entry['num_events'], new_event_count=new_entry['num_events'],
                          event_read=new_entry.get('event_read'), representations={})
            rows = []
            for rep in reps:
                old_path = folder / f'{stem}_{rep}.png'
                new_path = args.output_root / folder.name / f'{stem}_{rep}.png'
                with Image.open(old_path) as im:
                    old = im.convert('RGB')
                with Image.open(new_path) as im:
                    new = im.convert('RGB')
                if old.size != new.size:
                    raise ValueError(f'Image size mismatch: {old_path}')
                difference = np.abs(np.asarray(old).astype(np.int16)-np.asarray(new).astype(np.int16))
                result['representations'][rep] = dict(
                    exact_match=bool(not difference.any()),
                    mean_absolute_error=float(difference.mean()), max_absolute_error=int(difference.max()),
                    changed_pixel_fraction=float(np.any(difference, axis=-1).mean()),
                    reference=str(old_path), indexed=str(new_path))
                diff = Image.fromarray(np.minimum(difference*8, 255).astype(np.uint8))
                rows.append([panel(old, f'{rep}: existing'), panel(new, f'{rep}: indexed'),
                             panel(diff, f'{rep}: absolute difference x8')])
                if rep == 'cstr3':
                    cx, cy, w, h = max(boxes, key=lambda box: box[2]*box[3])
                    # Context around the largest annotated UAV, identical in both.
                    w, h = max(w*2, 0.10), max(h*2, 0.16)
                    roi = (max(0, int((cx-w/2)*old.width)), max(0, int((cy-h/2)*old.height)),
                           min(old.width, int((cx+w/2)*old.width)), min(old.height, int((cy+h/2)*old.height)))
                    zooms = [im.crop(roi).resize((480, 270), Image.Resampling.NEAREST) for im in (old, new, diff)]
                    rows.append([panel(im, title) for im, title in zip(zooms,
                                 ['UAV crop: existing', 'UAV crop: indexed', 'UAV crop: difference x8'])])
            sheet = Image.new('RGB', (1440, len(rows)*300))
            for y, row in enumerate(rows):
                for x, im in enumerate(row):
                    sheet.paste(im, (x*480, y*300))
            sheet_path = preview_dir / f'{folder.name}_{stem}.jpg'
            sheet.save(sheet_path, quality=95)
            result['comparison'] = str(sheet_path)
            report['frames'].append(result)
            relative = sheet_path.relative_to(args.output_root).as_posix()
            originals = ' | '.join(
                f'<a href="{html.escape(os.path.relpath(Path(result["representations"][r][kind]), args.output_root).replace(chr(92), "/"))}">{r} {kind}</a>'
                for r in reps for kind in ('reference', 'indexed'))
            links.append(f'<h2>{html.escape(folder.name+" / "+stem)}</h2><p>{originals}</p>'
                         f'<a href="{relative}"><img loading="lazy" src="{relative}"></a>')
            print(f"{folder.name}/{stem}: events {entry['num_events']} -> {new_entry['num_events']}; "
                  + ', '.join(f"{r}: MAE={result['representations'][r]['mean_absolute_error']:.5f}" for r in reps), flush=True)
    (args.output_root/'comparison_results.json').write_text(json.dumps(report, indent=2))
    page = ('<!doctype html><meta charset="utf-8"><title>FRED indexed RAW comparison</title>'
            '<style>body{background:#10151e;color:#e5eaf2;font:16px system-ui;margin:28px}'
            'img{max-width:100%;height:auto}a{color:#8dcaff}h2{margin-top:45px}</style>'
            '<h1>Existing renders / indexed RAW renders</h1>'
            '<p>Random labeled UAV frames. Each sheet: existing, indexed, difference amplified 8×. '
            'Rows: CSTR3, UAV crop, XT, YT. Original PNGs are linked; JPEG sheets are previews only.</p>'
            + ''.join(links))
    (args.output_root/'index.html').write_text(page, encoding='utf-8')


if __name__ == '__main__':
    main()
