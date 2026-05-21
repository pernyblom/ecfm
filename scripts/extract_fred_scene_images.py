import argparse
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg"}
RGB_DIR_NAMES = ("RGB", "PADDED_RGB")


def _iter_scene_dirs(fred_root: Path) -> list[Path]:
    scene_dirs = [path for path in fred_root.iterdir() if path.is_dir() and path.name.isdigit()]
    return sorted(
        scene_dirs,
        key=lambda path: (
            not path.name.isdigit(),
            int(path.name) if path.name.isdigit() else path.name,
        ),
    )


def _iter_rgb_images(rgb_dir: Path) -> list[Path]:
    images = [
        path
        for path in rgb_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(images, key=lambda path: path.name)


def _find_rgb_dir(scene_dir: Path) -> Path | None:
    for name in RGB_DIR_NAMES:
        rgb_dir = scene_dir / name
        if rgb_dir.is_dir():
            return rgb_dir
    return None


def extract_scene_images(fred_root: Path, output_dir: Path, overwrite: bool) -> tuple[int, int]:
    if not fred_root.exists():
        raise FileNotFoundError(f"FRED root does not exist: {fred_root}")
    if not fred_root.is_dir():
        raise NotADirectoryError(f"FRED root is not a directory: {fred_root}")

    output_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    skipped = 0

    for scene_dir in _iter_scene_dirs(fred_root):
        rgb_dir = _find_rgb_dir(scene_dir)
        if rgb_dir is None:
            print(
                f"Skipping {scene_dir.name}: missing RGB directory "
                f"({', '.join(RGB_DIR_NAMES)})."
            )
            skipped += 1
            continue

        images = _iter_rgb_images(rgb_dir)
        if not images:
            print(f"Skipping {scene_dir.name}: no JPEG images in {rgb_dir}.")
            skipped += 1
            continue

        output_path = output_dir / f"{scene_dir.name}.jpg"
        if output_path.exists() and not overwrite:
            print(f"Skipping {scene_dir.name}: {output_path} already exists.")
            skipped += 1
            continue

        middle_image = images[len(images) // 2]
        shutil.copy2(middle_image, output_path)
        copied += 1
        print(f"{scene_dir.name}: {middle_image.name} -> {output_path}")

    return copied, skipped


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract the middle RGB frame from each FRED scene into a flat overview "
            "folder named by scene ID."
        )
    )
    parser.add_argument(
        "--fred-root",
        type=Path,
        default=Path("datasets/FRED"),
        help="Root path of FRED dataset (default: datasets/FRED).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/scene_images"),
        help="Output directory for scene overview images (default: outputs/scene_images).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output images.",
    )
    args = parser.parse_args()

    copied, skipped = extract_scene_images(args.fred_root, args.output_dir, args.overwrite)
    print(f"Copied {copied} scene images. Skipped {skipped}.")


if __name__ == "__main__":
    main()
