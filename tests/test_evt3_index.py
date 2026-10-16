import json
from pathlib import Path

import numpy as np
import pytest

from ecfm.utils.evt3 import _decode_evt3_words
from ecfm.utils.evt3_index import INDEX_DTYPE, INDEX_MAGIC, IndexedRawReader
from ecfm.utils.file_io import retry_file_io
from scripts.render_evt3_yolo_frames import build_parser, ensure_rendered_frames, render_yolo_frames


def recording(tmp_path, times=None, shift=16773120):
    times = times or [16774000, 16775000, 16777215, 16777216, 16777217, 16780001, 16785003, 16790005]
    header = b'% format EVT3;height=8;width=8\n% end\n'
    words = []
    offsets = []
    for i, t in enumerate(times):
        offsets.append(len(header) + 2*len(words))
        words.extend([0x8000 | ((t >> 12) & 0xFFF), 0x6000 | (t & 0xFFF),
                      i % 8, 0x2800 | (i % 8)])
    raw = tmp_path / 'events.raw'
    raw.write_bytes(header + np.asarray(words, dtype='<u2').tobytes())
    local = np.asarray(times) - shift
    records = []
    for slot in range(int(local[-1] // 2000) + 2):
        previous = np.flatnonzero(local <= slot * 2000)
        j = int(previous[-1]) if len(previous) else None
        records.append((-1, len(header), 0) if j is None else (int(local[j]), offsets[j], 1))
    index_header = (f'% format EVT3;height=8;width=8\n% index_version 2.0\n'
                    f'% size {raw.stat().st_size}\n% ts_shift_us {shift}\n'
                    '% bookmark_period_us 2000\n% end\n').encode()
    raw.with_suffix('.raw.tmp_index').write_bytes(index_header + np.array(records, dtype=INDEX_DTYPE).tobytes() + INDEX_MAGIC)
    return raw, np.asarray(words, dtype='<u2'), shift


@pytest.mark.parametrize('mode', ['integer', 'legacy_float32'])
def test_indexed_matches_sequential_at_wraps_and_bounds(tmp_path, mode):
    raw, words, shift = recording(tmp_path)
    dtype = np.int64 if mode == 'integer' else np.float32
    reference, _, _ = _decode_evt3_words(words, output_dtype=dtype)
    reference[:, 2] = reference[:, 2].astype(np.float64) - shift
    reader = IndexedRawReader(raw, timestamp_mode=mode, chunk_bytes=6, guard_us=0)
    # Deliberately nonchronological requests; includes prefix fallback, EOF and quiet intervals.
    for start, end in [(10000, 18000), (0, 2000), (4095, 4098), (6881, 6882),
                       (19000, 20000), (9000, 10000), (1000, 1000)]:
        times = reference[:, 2].astype(np.float64)
        expected = reference[(times >= start) & (times < end)]
        np.testing.assert_array_equal(reader.read_window(start, end), expected)
    assert IndexedRawReader(raw, guard_us=0).read_window(6881, 6882)[0, 2] == 6881


def test_legacy_unrepresentable_window_bound(tmp_path):
    raw, words, _ = recording(tmp_path, times=[4096, 16777215, 16777216, 33554431,
                                             33554432, 49132840, 49132844], shift=0)
    reference, _, _ = _decode_evt3_words(words)
    reader = IndexedRawReader(raw, timestamp_mode='legacy_float32', guard_us=0)
    start, end = 49132839, 49132842
    expected = reference[np.searchsorted(reference[:, 2], start):np.searchsorted(reference[:, 2], end)]
    actual = reader.read_window(start, end)
    np.testing.assert_array_equal(actual, expected)
    assert len(actual) == 1


def test_rejects_stale_and_incomplete_indexes(tmp_path):
    raw, _, _ = recording(tmp_path)
    index = raw.with_suffix('.raw.tmp_index')
    original = index.read_bytes()
    index.write_bytes(original[:-1])
    with pytest.raises(ValueError, match='incomplete'):
        IndexedRawReader(raw)
    index.write_bytes(original)
    raw.write_bytes(raw.read_bytes()+b'\x00\x00')
    with pytest.raises(ValueError, match='size'):
        IndexedRawReader(raw)


@pytest.mark.parametrize('target', ['raw', 'index'])
def test_index_loading_retries_temporary_locks(tmp_path, monkeypatch, target):
    raw, _, _ = recording(tmp_path)
    locked = raw if target == 'raw' else raw.with_suffix('.raw.tmp_index')
    original_open = Path.open
    calls = 0

    def open_file(path, *args, **kwargs):
        nonlocal calls
        if path == locked:
            calls += 1
            if calls <= 2:
                raise PermissionError(13, 'simulated scanner lock', str(path))
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', open_file)
    monkeypatch.setattr('ecfm.utils.file_io.time.sleep', lambda _: None)
    with pytest.warns(RuntimeWarning, match='retrying'):
        reader = IndexedRawReader(raw)
    assert len(reader.read_window(0, 18000)) > 0
    assert calls >= 3


def test_index_short_read_reports_truncation(tmp_path, monkeypatch):
    raw, _, _ = recording(tmp_path)
    monkeypatch.setattr('ecfm.utils.evt3_index.np.fromfile',
                        lambda *args, **kwargs: np.empty(0, dtype=INDEX_DTYPE))
    with pytest.raises(ValueError, match='Index changed or was truncated'):
        IndexedRawReader(raw)


def test_window_retry_restarts_decoder_after_partial_read(tmp_path, monkeypatch):
    raw, _, _ = recording(tmp_path)
    reader = IndexedRawReader(raw, chunk_bytes=8)
    expected = reader.read_window(0, 18000)
    expected_stats = reader.last_read.copy()
    original_open = Path.open
    streams = []

    class InterruptedStream:
        def __init__(self, stream):
            self.stream = stream
            self.reads = 0

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.stream.close()

        def seek(self, offset):
            return self.stream.seek(offset)

        def read(self, size):
            self.reads += 1
            if self.reads == 2:
                raise PermissionError(13, 'simulated lock during read', str(raw))
            return self.stream.read(size)

    def open_file(path, *args, **kwargs):
        stream = original_open(path, *args, **kwargs)
        streams.append(stream)
        return InterruptedStream(stream) if len(streams) == 1 else stream

    monkeypatch.setattr(Path, 'open', open_file)
    monkeypatch.setattr('ecfm.utils.file_io.time.sleep', lambda _: None)
    with pytest.warns(RuntimeWarning, match='retrying'):
        actual = reader.read_window(0, 18000)
    np.testing.assert_array_equal(actual, expected)
    assert reader.last_read == expected_stats
    assert len(streams) == 2 and all(s.closed for s in streams)


@pytest.mark.parametrize('error, attempts', [
    (PermissionError(13, 'persistent lock'), 8),
    (FileNotFoundError(2, 'missing'), 1),
    (OSError(5, 'disk I/O error'), 1),
    (ValueError('corrupt index'), 1),
])
def test_file_retries_are_bounded_and_selective(monkeypatch, error, attempts):
    calls, delays = [], []

    def fail():
        calls.append(1)
        raise error

    monkeypatch.setattr('ecfm.utils.file_io.time.sleep', delays.append)
    if attempts > 1:
        with pytest.warns(RuntimeWarning), pytest.raises(type(error)):
            retry_file_io(fail, description='test file')
    else:
        with pytest.raises(type(error)):
            retry_file_io(fail, description='test file')
    assert len(calls) == attempts
    assert sum(delays) == pytest.approx(4.5 if attempts > 1 else 0)


def test_missing_representation_regenerated_and_hits_do_not_decode(tmp_path, monkeypatch):
    raw, _, _ = recording(tmp_path)
    labels = tmp_path / 'labels'
    labels.mkdir()
    (labels/'Video_0_frame_18000.txt').write_text('0 0.5 0.5 0.4 0.4\n')
    # Another label must not be rendered by a targeted request.
    (labels/'Video_0_frame_10000.txt').write_text('0 0.5 0.5 0.4 0.4\n')
    output = tmp_path / 'images'
    args = build_parser().parse_args([str(raw), str(labels), str(output), '--event-source', 'indexed',
                                    '--representation', 'cstr3;xt_my', '--window', '18000',
                                    '--label-stems', 'Video_0_frame_18000'])
    render_yolo_frames(args)
    manifest = json.loads((output/'render_manifest.json').read_text())
    params = manifest['render_params']
    paths = [Path(manifest['files'][0]['representations'][r]['path']) for r in ['cstr3', 'xt_my']]
    expected = paths[0].read_bytes()
    untouched = paths[1].stat().st_mtime_ns
    paths[0].unlink()
    ensure_rendered_frames(raw, labels, output, ['Video_0_frame_18000'], params)
    assert paths[0].read_bytes() == expected
    assert paths[1].stat().st_mtime_ns == untouched
    assert not (output/'Video_0_frame_10000_cstr3.png').exists()
    def unexpected(*args, **kwargs):
        raise AssertionError('Cache hit must not decode events')
    monkeypatch.setattr(IndexedRawReader, 'read_window', unexpected)
    ensure_rendered_frames(raw, labels, output, ['Video_0_frame_18000'], params)
    with pytest.raises(ValueError, match='different render parameters'):
        ensure_rendered_frames(raw, labels, output, ['Video_0_frame_18000'], dict(params, window=10000))
