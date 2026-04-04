import shutil
import uuid
from pathlib import Path

from experiments.pose_dynamics.data.track_projection_dataset import TrackProjectionDataset


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_png_stub(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x89PNG\r\n\x1a\n")


def _make_test_root() -> Path:
    root = Path.cwd() / "outputs" / "test_tmp" / f"pose_dynamics_cache_{uuid.uuid4().hex}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _build_dataset(tmp_path: Path) -> TrackProjectionDataset:
    images_root = tmp_path / "images"
    labels_root = tmp_path / "labels"
    folder = "seq_001"
    labels_dir = labels_root / folder / "Event_YOLO"
    images_dir = images_root / folder

    _write_text(labels_dir / "cam_frame_000001.txt", "frame\n")
    _write_text(labels_dir / "cam_frame_000002.txt", "frame\n")
    _write_text(labels_root / folder / "cleaned_tracks.txt", "0.0,1,10,20,4,6\n1.0,1,12,22,4,6\n")
    _write_png_stub(images_dir / "cam_frame_000001_xt_my.png")
    _write_png_stub(images_dir / "cam_frame_000001_yt_mx.png")
    _write_png_stub(images_dir / "cam_frame_000002_xt_my.png")
    _write_png_stub(images_dir / "cam_frame_000002_yt_mx.png")

    return TrackProjectionDataset(
        images_root=images_root,
        labels_root=labels_root,
        representations=["xt_my", "yt_mx"],
        image_size=(32, 32),
        history_steps=0,
        future_steps=1,
        stride=1,
        frame_size=(1280, 720),
        intrinsics=(1.0, 1.0, 0.5, 0.5),
        camera_pose=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        folders=[folder],
        cache_dir=tmp_path / "cache",
    )


def test_cache_key_changes_when_track_file_changes():
    tmp_path = _make_test_root()
    try:
        dataset = _build_dataset(tmp_path)
        key_before = dataset._cache_key()

        tracks_path = tmp_path / "labels" / "seq_001" / "cleaned_tracks.txt"
        tracks_path.write_text("0.0,1,10,20,4,6\n2.0,1,12,22,4,6\n", encoding="utf-8")

        dataset_after = _build_dataset(tmp_path)
        key_after = dataset_after._cache_key()

        assert key_after != key_before
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_cache_key_changes_when_representation_availability_changes():
    tmp_path = _make_test_root()
    try:
        dataset = _build_dataset(tmp_path)
        key_before = dataset._cache_key()

        rep_path = tmp_path / "images" / "seq_001" / "cam_frame_000002_yt_mx.png"
        rep_path.unlink()

        dataset_after = _build_dataset(tmp_path)
        key_after = dataset_after._cache_key()

        assert key_after != key_before
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)
