"""Compare middle RGB frames and create contact sheets for scene review."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def middle_frames(root: Path) -> dict[str, Path]:
    frames = {}
    for folder in sorted((p for p in root.iterdir() if p.is_dir() and p.name.isdigit()),
                         key=lambda p: int(p.name)):
        for kind in ("RGB", "PADDED_RGB"):
            directory = folder / kind
            if not directory.is_dir():
                continue
            with os.scandir(directory) as entries:
                rgb = sorted(p.name for p in entries if p.is_file()
                             and p.name.lower().endswith((".jpg", ".jpeg", ".png"))
                             and not p.name.startswith("."))
            if rgb:
                frames[folder.name] = directory / rgb[len(rgb) // 2]
                break
    return frames


def contact_sheets(output: Path, frames: dict[str, Path], labels: dict[str, str]) -> None:
    items = list(frames)
    tile_w, tile_h, columns, rows = 256, 165, 5, 6
    per_page = columns * rows
    for offset in range(0, len(items), per_page):
        sheet = Image.new("RGB", (tile_w * columns, tile_h * rows), "#161c24")
        draw = ImageDraw.Draw(sheet)
        for index, key in enumerate(items[offset:offset + per_page]):
            x, y = index % columns * tile_w, index // columns * tile_h
            with Image.open(frames[key]) as image:
                image.thumbnail((tile_w - 4, tile_h - 22))
                sheet.paste(image, (x + 2, y + 2))
            draw.text((x + 6, y + tile_h - 19), labels[key], fill="white")
        sheet.save(output / f"sheet_{offset // per_page + 1:02d}.jpg", quality=88)


def match_frames(frames: dict[str, Path], neighbors: int):
    import cv2

    cv2.setNumThreads(2)
    sift = cv2.SIFT_create(nfeatures=700)
    matcher = cv2.BFMatcher()
    features, descriptors = [], []
    for path in frames.values():
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        gray = cv2.resize(gray, (640, 360), interpolation=cv2.INTER_AREA)
        thumbnail = cv2.resize(gray, (32, 18), interpolation=cv2.INTER_AREA).astype(np.float32)
        thumbnail -= thumbnail.mean()
        descriptors.append((thumbnail / max(1.0, np.linalg.norm(thumbnail))).ravel())
        features.append(sift.detectAndCompute(gray, None))
    vectors = np.asarray(descriptors)
    similarities = vectors @ vectors.T
    pairs = set()
    for i in range(len(frames)):
        candidates = np.argsort(-similarities[i])[:neighbors + 1].tolist()
        candidates.extend(range(max(0, i - 4), min(len(frames), i + 5)))
        pairs.update(tuple(sorted((i, j))) for j in candidates if i != j)
    ids = list(frames)
    corners = np.float32([[[0, 0], [639, 0], [639, 359], [0, 359]]])
    for i, j in sorted(pairs):
        points_a, desc_a = features[i]
        points_b, desc_b = features[j]
        inliers, displacement = 0, None
        if desc_a is not None and desc_b is not None and min(len(desc_a), len(desc_b)) >= 2:
            matches = []
            for pair in matcher.knnMatch(desc_a, desc_b, k=2):
                if len(pair) == 2 and pair[0].distance < 0.73 * pair[1].distance:
                    matches.append(pair[0])
            if len(matches) >= 8:
                a = np.float32([points_a[m.queryIdx].pt for m in matches])
                b = np.float32([points_b[m.trainIdx].pt for m in matches])
                affine, mask = cv2.estimateAffinePartial2D(a, b, ransacReprojThreshold=3.0)
                if affine is not None:
                    inliers = int(mask.sum())
                    delta = cv2.transform(corners, affine) - corners
                    displacement = round(float(np.linalg.norm(delta, axis=-1).mean()), 2)
        yield ids[i], ids[j], round(float(similarities[i, j]), 4), inliers, displacement


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("D:/Datasets/FRED"))
    parser.add_argument("--output", type=Path, default=Path("outputs/fred_relative_pose/scene_review"))
    parser.add_argument("--neighbors", type=int, default=12)
    parser.add_argument("--manifest", type=Path, help="Order sheets by the reviewed scene IDs")
    args = parser.parse_args()
    frames = middle_frames(args.root)
    if not frames:
        parser.error(f"No RGB sequences found in {args.root}")
    args.output.mkdir(parents=True, exist_ok=True)
    labels = {key: key for key in frames}
    if args.manifest:
        import yaml

        scenes = yaml.safe_load(args.manifest.read_text(encoding="utf-8"))["scenes"]
        labels.update({str(key): f"{key}  {scene}" for scene, ids in scenes.items() for key in ids})
        frames = dict(sorted(frames.items(), key=lambda item: (labels[item[0]].partition("  ")[2], int(item[0]))))
    contact_sheets(args.output, frames, labels)
    with (args.output / "frames.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sequence", "image", "cleaned_tracks"])
        writer.writerows((key, path.relative_to(args.root), (args.root / key / "cleaned_tracks.txt").is_file())
                         for key, path in frames.items())
    if args.neighbors > 0:
        with (args.output / "similarity.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["sequence_a", "sequence_b", "thumbnail_similarity", "sift_inliers", "corner_shift_px_at_640"])
            writer.writerows(match_frames(frames, args.neighbors))


if __name__ == "__main__":
    main()
