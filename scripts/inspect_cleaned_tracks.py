"""Render cleaned FRED tracks directly, without forecasting or time realignment."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import re
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.kalman_ml_forecasting.compose_track_layers import write_mp4_iter
from experiments.kalman_ml_forecasting.data.track_dataset import _read_tracks


def font(size):
    try:
        return ImageFont.truetype("C:/Windows/Fonts/arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def rgb_index(directory):
    result = []
    for path in directory.iterdir():
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png"} or path.name.startswith("."):
            continue
        match = re.search(r"_(\d{2})_(\d{2})_(\d{2})\.(\d+)$", path.stem)
        if match:
            hh, mm, ss, frac = match.groups()
            result.append((3600 * int(hh) + 60 * int(mm) + int(ss) + float("0." + frac), path))
    result.sort()
    if not result:
        raise ValueError(f"No timestamped RGB frames in {directory}")
    base = result[0][0]
    return np.array([t - base for t, _ in result]), [p for _, p in result]


def choose_tracks(tracks):
    # Show a long track, a typical usable track, and a possible discontinuity.
    eligible = [(tid, np.asarray(rows, dtype=float)) for tid, rows in tracks.items()
                if len(rows) > 1 and rows[-1][0] - rows[0][0] >= 1.2]
    if not eligible:
        raise ValueError("No cleaned tracks last at least 1.2 seconds")
    eligible.sort(key=lambda item: (item[1][-1, 0] - item[1][0, 0], item[0]))
    chosen = [("longest", eligible[-1])]
    remaining = [item for item in eligible if item[0] != chosen[0][1][0]]
    if remaining:
        chosen.append(("median usable duration", remaining[len(remaining) // 2]))
    used = {item[1][0] for item in chosen}
    remaining = [item for item in eligible if item[0] not in used]
    if remaining:
        def jump(item):
            rows = item[1]
            centers = rows[:, 1:3] + rows[:, 3:5] / 2
            return float(np.linalg.norm(np.diff(centers, axis=0), axis=1).max())
        chosen.append(("largest frame-to-frame jump", max(remaining, key=jump)))
    return chosen


def render(rows, row_index, times, paths, folder, tid, reason):
    t, x, y, w, h = rows[row_index]
    frame_index = int(np.argmin(np.abs(times - t)))
    with Image.open(paths[frame_index]) as source:
        source = source.convert("RGB")
    annotated = source.copy()
    draw = ImageDraw.Draw(annotated)
    history = rows[:row_index + 1]
    history = history[history[:, 0] >= t - 1.0]
    centers = history[:, 1:3] + history[:, 3:5] / 2
    if len(centers) > 1:
        draw.line([tuple(p) for p in centers], fill="#30d5ff", width=3)
    draw.rectangle((x, y, x + w, y + h), outline="#00ff70", width=3)
    cx, cy = x + w / 2, y + h / 2
    # Fixed crop per track avoids apparent zoom changes as the box size changes.
    side = max(160, int(np.percentile(np.maximum(rows[:, 3], rows[:, 4]), 95) * 3))
    bounds = (round(cx - side / 2), round(cy - side / 2),
              round(cx - side / 2) + side, round(cy - side / 2) + side)
    zoom = annotated.crop(bounds).resize((480, 480), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (1440, 640), "#111827")
    canvas.paste(annotated.resize((960, 540), Image.Resampling.LANCZOS), (0, 90))
    canvas.paste(zoom, (960, 120))
    draw = ImageDraw.Draw(canvas)
    draw.text((14, 8), f"Folder {folder}  |  cleaned track {tid}  |  {reason}", font=font(24), fill="white")
    draw.text((14, 43), f"t={t:.3f}s   RGB offset={times[frame_index] - t:+.3f}s   green: box   cyan: previous 1s", font=font(21), fill="#cbd5e1")
    draw.text((974, 94), f"Object close-up ({side} x {side} source pixels)", font=font(19), fill="white")
    return canvas, paths[frame_index], float(times[frame_index] - t)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("D:/datasets/FRED"))
    parser.add_argument("--folders", nargs="+", default=["22", "67", "101", "113"])
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/cleaned_tracks_inspection"))
    parser.add_argument("--max-seconds", type=float, default=6.0)
    parser.add_argument("--fps", type=float, default=15.0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records, sections = [], []
    for folder in args.folders:
        directory = args.root / folder
        tracks = _read_tracks(directory / "cleaned_tracks.txt")
        times, paths = rgb_index(directory / "PADDED_RGB")
        selected = choose_tracks(tracks)
        sheet = Image.new("RGB", (1920, 80 + 310 * len(selected)), "#111827")
        ImageDraw.Draw(sheet).text((16, 16), f"Folder {folder} - cleaned tracks on PADDED_RGB (green box; cyan trail)", font=font(30), fill="white")
        cards = []
        for track_num, (reason, (tid, rows)) in enumerate(selected):
            start, end = float(rows[0, 0]), float(rows[-1, 0])
            gaps = np.diff(rows[:, 0])
            centers = rows[:, 1:3] + rows[:, 3:5] / 2
            jumps = np.linalg.norm(np.diff(centers, axis=0), axis=1)
            # Center the diagnostic clip on the jump for that selection.
            clip_start = start
            if reason.startswith("largest"):
                clip_start = max(start, min(end - args.max_seconds, float(rows[int(jumps.argmax()), 0]) - args.max_seconds / 2))
            clip_end = min(end, clip_start + args.max_seconds)
            query_times = np.arange(clip_start, clip_end + 1e-9, 1 / args.fps)
            indices = [int(np.argmin(np.abs(rows[:, 0] - t))) for t in query_times]
            name = f"folder_{folder}_track_{tid}.mp4"
            offsets = []
            def frames():
                for index in indices:
                    frame, _, offset = render(rows, index, times, paths, folder, tid, reason)
                    offsets.append(offset)
                    yield frame
            write_mp4_iter(frames(), args.output_dir / name, fps=args.fps)
            stills = []
            for column, query in enumerate(np.linspace(clip_start, clip_end, 4)):
                index = int(np.argmin(np.abs(rows[:, 0] - query)))
                frame, path, offset = render(rows, index, times, paths, folder, tid, reason)
                # Keep a full scene plus an unobstructed close-up in every panel.
                panel = Image.new("RGB", (480, 310), "#111827")
                scene = frame.crop((0, 90, 960, 630)).resize((480, 270))
                panel.paste(scene, (0, 40))
                panel.paste(frame.crop((960, 120, 1440, 600)).resize((165, 165)), (315, 145))
                ImageDraw.Draw(panel).text((5, 5), f"ID {tid}   t={rows[index, 0]:.2f}s", font=font(22), fill="white")
                sheet.paste(panel, (column * 480, 80 + track_num * 310))
                stills.append({"time_s": float(rows[index, 0]), "rgb": str(path), "rgb_offset_s": offset})
            record = {"folder": folder, "track_id": tid, "selection": reason, "rows": len(rows),
                      "start_s": start, "end_s": end, "duration_s": end - start,
                      "median_step_s": float(np.median(gaps)), "max_gap_s": float(gaps.max()),
                      "max_center_jump_px": float(jumps.max()), "clip_start_s": clip_start,
                      "clip_end_s": clip_end, "max_rgb_offset_s": float(np.max(np.abs(offsets))),
                      "video": name, "stills": stills}
            records.append(record)
            cards.append(f'<article><h3>Track {tid}: {html.escape(reason)}</h3><p>{len(rows)} points; {end-start:.2f}s; largest step {jumps.max():.1f}px. Clip {clip_start:.2f}-{clip_end:.2f}s.</p><video controls loop preload="metadata" src="{name}"></video></article>')
            print(f"Folder {folder}, track {tid}: {reason}; {len(rows)} points; {end-start:.2f}s; RGB error <= {max(map(abs, offsets)):.3f}s", flush=True)
        sheet_name = f"folder_{folder}_contact_sheet.jpg"
        sheet.save(args.output_dir / sheet_name, quality=93)
        sections.append(f'<section id="folder-{folder}"><h2>Folder {folder}</h2><p>{len(tracks)} cleaned tracks in file. Three duration-eligible samples shown.</p><a href="{sheet_name}"><img class="sheet" src="{sheet_name}"></a>{"".join(cards)}</section>')
    (args.output_dir / "inspection.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    report = '''<!doctype html><meta charset="utf-8"><title>Cleaned track inspection</title>
<style>body{background:#0f172a;color:#e2e8f0;font:17px system-ui;max-width:1400px;margin:32px auto;padding:0 24px}a{color:#67e8f9}h2{margin-top:48px}video,.sheet{width:100%;border-radius:8px}article{padding:16px;background:#1e293b;margin:18px 0;border-radius:12px}p{line-height:1.5}</style>
<h1>Cleaned track inspection</h1><p>Green: current cleaned bounding box. Cyan: last second of track centers. Right: object close-up. Clips play at real time, sampled at 15 fps. These are existing cleaned rows, with no track-time shifts or box interpolation.</p>
<p>PADDED_RGB frames use elapsed time from the first camera timestamp, matching the forecasting loader. RGB offset is shown on every video frame. The green boxes originate in event-camera annotations; a small camera alignment offset may remain. A few examples cannot establish the quality of an entire folder.</p>
<nav>Folders: <a href="#folder-22">22</a> | <a href="#folder-67">67</a> | <a href="#folder-101">101</a> | <a href="#folder-113">113</a></nav>'''
    (args.output_dir / "index.html").write_text(report + "".join(sections), encoding="utf-8")


if __name__ == "__main__":
    main()
