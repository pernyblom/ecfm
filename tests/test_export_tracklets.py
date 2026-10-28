from copy import deepcopy
import csv
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.kalman_ml_forecasting.export_tracklets import export_tracklets
from experiments.kalman_ml_forecasting.train import _split_train_eval_folders


def _sequence(root, folder, *, track_ids=(7,), times_us=None, original=True):
    if times_us is None:
        times_us = list(range(0, 2_000_001, 100_000))
    labels = root / folder / "Event_YOLO"
    labels.mkdir(parents=True)
    for t in times_us:
        (labels / f"Video_0_frame_{t}.txt").write_text("", encoding="utf-8")
    rows = [f"{t / 1e6:.6f},{track_id},{10 + t / 1e6},20,4,6"
            for track_id in track_ids for t in times_us]
    if original:
        (root / folder / "tracks.txt").write_text("\n".join(rows), encoding="utf-8")
    # Deliberately different data: the exporter must never read these tracks.
    (root / folder / "cleaned_tracks.txt").write_text(
        "0,999,90,90,2,2\n2,999,90,90,2,2\n", encoding="utf-8")


def _config(tmp_path, folders):
    split = tmp_path / "split.txt"
    split.write_text("\n".join(folders), encoding="utf-8")
    return {"data": {
        "labels_root": str(tmp_path / "labels"),
        "frame_size": [100, 100],
        "split_files": {"train": str(split), "test": str(split)},
        "time_align": "none",
        "seed": 42,
        "history_ms": 400,
        "forecast_ms": 800,
        "tracks_file": "cleaned_tracks.txt",
        # These settings must not cause image reads, cache reuse or augmentation.
        "representations": ["cstr3"],
        "event_image_source": "indexed",
        "cache_dir": str(tmp_path / "no_cache"),
        "instance_cache": {"enabled": True, "mode": "eager"},
        "box_augmentation": {"enabled": True, "cache": True},
    }}


def _rows(output_dir):
    return [json.loads(line) for line in (output_dir / "tracklets.jsonl").read_text().splitlines()]


def test_original_tracks_only_and_collision_ids_reserve_unique_originals(tmp_path):
    cfg = _config(tmp_path, ["a", "b", "cleaned_only", "missing", "a"])
    root = Path(cfg["data"]["labels_root"])
    _sequence(root, "a", track_ids=(7, 8))
    _sequence(root, "b", track_ids=(7, 9))
    _sequence(root, "cleaned_only", original=False)
    before = deepcopy(cfg)
    output = tmp_path / "export"
    manifest = export_tracklets(cfg, split="train", output_dir=output)
    rows = _rows(output)

    assert manifest["skipped_folders_without_tracks_txt"] == ["cleaned_only", "missing"]
    assert manifest["eligible_folders"] == ["a", "b"]
    assert manifest["tracks"] == 4
    assert len(rows) == 36  # Nine overlapping 1.2 s windows per original track.
    mapping = {(row["folder"], row["original_track_id"]): row["track_id"] for row in rows}
    assert mapping == {("a", 7): 7, ("a", 8): 8, ("b", 7): 10, ("b", 9): 9}
    assert len({row["tracklet_id"] for row in rows}) == len(rows)
    assert all(row["duration_s"] >= 1.2 for row in rows)
    assert all(row["source_tracks_file"] == f"{row['folder']}/tracks.txt" for row in rows)
    assert all(len(row["past_times_s"]) == len(row["past_boxes"]) == 5 for row in rows)
    assert all(len(row["future_times_s"]) == len(row["future_boxes"]) == 8 for row in rows)
    np.testing.assert_allclose(rows[0]["past_boxes"][0], [0.12, 0.23, 0.04, 0.06])
    assert not (tmp_path / "no_cache").exists()
    assert cfg == before
    assert json.loads((output / "manifest.json").read_text()) == manifest
    with (output / "tracklets.csv").open(newline="", encoding="utf-8") as stream:
        index = list(csv.DictReader(stream))
    assert [row["tracklet_id"] for row in index] == [row["tracklet_id"] for row in rows]
    assert all(float(row["duration_s"]) >= 1.2 for row in index)


def test_actual_duration_filter_precedes_caps_and_uses_full_precision(tmp_path):
    cfg = _config(tmp_path, ["seq"])
    cfg["data"]["label_period_s"] = 0.1
    cfg["data"]["max_samples_train"] = 1
    # At a large absolute timestamp, float32 loses the microsecond differences.
    offset = 100_000_000_000
    times = [offset + t for t in range(0, 1_200_000, 100_000)]
    times += [offset + 1_199_000, offset + 1_300_000]
    _sequence(Path(cfg["data"]["labels_root"]), "seq", times_us=times)
    output = tmp_path / "export"
    manifest = export_tracklets(cfg, split="train", output_dir=output)
    rows = _rows(output)

    assert len(rows) == 1
    assert manifest["skipped_short_windows"] == 1
    assert rows[0]["duration_s"] == 1.2
    assert rows[0]["past_times_s"][0] == pytest.approx(100000.1)
    assert np.diff(rows[0]["past_times_s"]).min() > 0.09


@pytest.mark.parametrize("forecast_ms", [200, 800])
def test_fred_cadence_keeps_36_intervals_with_timestamp_rounding(tmp_path, forecast_ms):
    cfg = _config(tmp_path, ["full", "short"])
    cfg["data"]["forecast_ms"] = forecast_ms
    root = Path(cfg["data"]["labels_root"])
    _sequence(root, "full", times_us=[i * 33333 for i in range(37)])
    _sequence(root, "short", times_us=[i * 33333 for i in range(36)])
    output = tmp_path / "export"
    export_tracklets(cfg, split="train", output_dir=output)
    rows = _rows(output)

    assert len(rows) == 1
    assert rows[0]["folder"] == "full"
    assert rows[0]["duration_s"] == 1.199988
    assert rows[0]["num_intervals"] == 36
    assert len(rows[0]["label_stems"]) == 37
    assert len(rows[0]["past_times_s"]) == 13
    assert len(rows[0]["future_times_s"]) == 24


def test_seeded_selection_decorrelation_and_tracklet_identity(tmp_path):
    cfg = _config(tmp_path, ["seq"])
    _sequence(Path(cfg["data"]["labels_root"]), "seq")
    cfg["data"]["max_samples_train"] = 7
    cfg["data"]["decorrelation"] = {
        "enabled": True, "splits": ["train"], "method": "random", "target_samples": 3,
        "min_samples": 1,
    }
    first, second, other, full = [tmp_path / name for name in ("first", "second", "other", "full")]
    export_tracklets(cfg, split="train", output_dir=first, seed=1)
    export_tracklets(cfg, split="train", output_dir=second, seed=1)
    export_tracklets(cfg, split="train", output_dir=other, seed=2)
    assert (first / "tracklets.jsonl").read_bytes() == (second / "tracklets.jsonl").read_bytes()
    assert len(_rows(first)) == 3
    assert {row["tracklet_id"] for row in _rows(first)} != {row["tracklet_id"] for row in _rows(other)}
    cfg["data"].pop("max_samples_train")
    cfg["data"]["decorrelation"]["enabled"] = False
    export_tracklets(cfg, split="train", output_dir=full)
    by_id = {row["tracklet_id"]: row for row in _rows(full)}
    assert all(row == by_id[row["tracklet_id"]] for row in _rows(first))


def test_training_holdout_and_split_seed_offset_match_training(tmp_path):
    cfg = _config(tmp_path, ["a", "b", "c", "d"])
    for folder in ["a", "b", "c", "d"]:
        _sequence(Path(cfg["data"]["labels_root"]), folder)
    cfg["data"]["train_eval_split"] = {"enabled": True, "eval_count": 1, "seed": 19}
    train_folders, eval_folders = _split_train_eval_folders(cfg)
    train = export_tracklets(cfg, split="train", output_dir=tmp_path / "train")
    evaluation = export_tracklets(cfg, split="train_eval", output_dir=tmp_path / "eval")
    test = export_tracklets(cfg, split="test", output_dir=tmp_path / "test", seed=50)

    assert train["eligible_folders"] == train_folders
    assert evaluation["eligible_folders"] == eval_folders
    assert set(train["exported_folders"]).isdisjoint(evaluation["exported_folders"])
    assert evaluation["dataset_seed"] == 43
    assert test["dataset_seed"] == 53


def test_track_cap_is_seeded_and_uses_original_ids(tmp_path):
    cfg = _config(tmp_path, ["seq"])
    _sequence(Path(cfg["data"]["labels_root"]), "seq", track_ids=(7, 8, 9, 10))
    cfg["data"]["max_tracks_test"] = 1
    output = tmp_path / "export"
    manifest = export_tracklets(cfg, split="test", output_dir=output, seed=123)
    keys = [("seq", track_id) for track_id in (7, 8, 9, 10)]
    np.random.default_rng(126).shuffle(keys)
    assert manifest["tracks"] == 1
    assert {row["original_track_id"] for row in _rows(output)} == {keys[0][1]}


def test_empty_export_and_unknown_split(tmp_path):
    cfg = _config(tmp_path, ["missing"])
    output = tmp_path / "export"
    manifest = export_tracklets(cfg, split="test", output_dir=output)
    assert manifest["tracklets"] == 0
    assert manifest["eligible_folders"] == []
    assert _rows(output) == []
    with pytest.raises(ValueError, match="data.split_files.typo"):
        export_tracklets(cfg, split="typo", output_dir=tmp_path / "typo")
