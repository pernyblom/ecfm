import json
from pathlib import Path

import numpy as np
import pytest

from ecfm.utils.evt3 import _decode_evt3_words
from ecfm.utils.evt3_index import INDEX_DTYPE, INDEX_MAGIC, IndexedRawReader
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
