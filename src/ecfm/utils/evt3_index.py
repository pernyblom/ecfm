"""Bounded event-window reads using existing little-endian OpenEB v2 indexes.

Reader instances own metadata only: each read opens its own RAW handle, so they
can be used with spawned workers without sharing a file cursor. Events are
[x, y, shifted timestamp in microseconds, polarity]. No decoded file is written.
"""
from pathlib import Path

import numpy as np

from .evt3 import _Evt3DecoderState, _decode_evt3_words, read_raw_header


INDEX_DTYPE = np.dtype([('t', '<i8'), ('offset', '<u8'), ('count', '<u4')])
_magic = np.random.RandomState(0x6D76).randint(0, 2**32, 20, dtype=np.uint32).astype(np.uint8).tobytes()
INDEX_MAGIC = _magic[8:16] + _magic[:8] + _magic[16:20]


def read_index(raw: Path):
    raw = Path(raw)
    sidecar = raw.with_suffix(raw.suffix + '.tmp_index')
    _, offset, meta = read_raw_header(sidecar)
    _, raw_offset, raw_meta = read_raw_header(raw)
    payload = sidecar.stat().st_size - offset
    if meta.get('index_version') != '2.0' or payload % 20 or payload < 40:
        raise ValueError(f'Unsupported/incomplete index layout: {sidecar}')
    if int(meta['size']) != raw.stat().st_size:
        raise ValueError('RAW size differs from index header')
    if raw_meta.get('format_name') != 'EVT3':
        raise ValueError('Indexed reader requires EVT3')
    if any(meta.get(k) != v for k, v in raw_meta.items()):
        raise ValueError('RAW header differs from index header')
    records = np.fromfile(sidecar, dtype=INDEX_DTYPE, offset=offset)
    if records[-1].tobytes() != INDEX_MAGIC:
        raise ValueError('Missing/unrecognized index completion marker')
    records = records[:-1]
    valid = records[records['t'] >= 0]
    if not len(valid) or np.any(np.diff(valid['t']) < 0):
        raise ValueError('Invalid index timestamp ordering')
    offsets = records['offset'].astype(np.int64)
    if (np.any(np.diff(offsets) < 0) or np.any(offsets < raw_offset)
            or np.any(offsets >= raw.stat().st_size)
            or np.any((offsets - raw_offset) % 2)):
        raise ValueError('Invalid RAW offsets')
    period, shift = int(meta['bookmark_period_us']), int(meta['ts_shift_us'])
    if period <= 0 or shift < 0:
        raise ValueError('Invalid index period or timestamp shift')
    return records, shift, period, raw_offset


def state_at(timestamp_us: int):
    high = timestamp_us >> 12
    return _Evt3DecoderState(
        time_low=timestamp_us & 0xFFF, time_high=high,
        time_high_raw=high & 0xFFF, time_high_base=high & ~0xFFF,
        last_time_high_raw=high & 0xFFF, last_time_low=timestamp_us & 0xFFF,
    )


class IndexedRawReader:
    def __init__(self, raw, *, timestamp_mode='integer', chunk_bytes=65536, guard_us=4000):
        self.raw = Path(raw)
        self.records, self.shift_us, self.period_us, self.data_offset = read_index(self.raw)
        self.metadata = read_raw_header(self.raw)[2]
        if timestamp_mode not in {'integer', 'legacy_float32'}:
            raise ValueError('timestamp_mode must be integer or legacy_float32')
        if chunk_bytes < 2 or chunk_bytes % 2 or guard_us < 0:
            raise ValueError('Need even chunk_bytes >= 2 and guard_us >= 0')
        self.timestamp_mode = timestamp_mode
        self.chunk_bytes, self.guard_us = chunk_bytes, guard_us
        self.last_read = {}

    def read_window(self, start_us, end_us):
        """Read [start_us,end_us); integer mode preserves exact microseconds.

        legacy_float32 intentionally reproduces the old RAW renderer's rounding
        before and after timestamp shifting, for compatibility with saved PNGs.
        """
        if not np.isfinite(start_us) or not np.isfinite(end_us) or end_us < start_us:
            raise ValueError('Window bounds must be finite and ordered')
        start_us = max(0, start_us)
        dtype = np.int64 if self.timestamp_mode == 'integer' else np.float32
        self.last_read = dict(bytes_read=0, offset=self.data_offset, bookmark_us=None)
        if end_us <= start_us:
            return np.empty((0, 4), dtype=dtype)
        slot = min(int(max(0, start_us-self.guard_us) // self.period_us), len(self.records)-1)
        bookmark = self.records[slot]
        if int(bookmark['t']) < 0 or int(bookmark['t']) > start_us:
            offset, state = self.data_offset, _Evt3DecoderState()
        else:
            offset = int(bookmark['offset'])
            state = state_at(int(bookmark['t']) + self.shift_us)
            self.last_read['bookmark_us'] = int(bookmark['t'])
        self.last_read['offset'] = offset
        chunks, counters = [], None
        # Extra clock margin covers float32 rounding in compatibility mode.
        margin = 0 if dtype == np.int64 else 2 * float(np.spacing(np.float32(end_us+self.shift_us)))
        with self.raw.open('rb') as stream:
            stream.seek(offset)
            while data := stream.read(self.chunk_bytes):
                if len(data) % 2:
                    raise ValueError('Truncated EVT3 word at end of RAW file')
                self.last_read['bytes_read'] += len(data)
                events, counters, state = _decode_evt3_words(
                    np.frombuffer(data, dtype='<u2'), state=state, counters=counters,
                    output_dtype=dtype)
                if dtype == np.int64:
                    events[:, 2] -= self.shift_us
                else:
                    events[:, 2] = (events[:, 2].astype(np.float64)-self.shift_us).astype(np.float32)
                if len(events) and np.any(np.diff(events[:, 2]) < 0):
                    raise ValueError('Nonmonotonic event timestamps in indexed window')
                # searchsorted in the old renderer compares against the exact
                # Python bound. A float32 boolean comparison can instead round
                # that scalar, dropping events just below an unrepresentable end.
                times = events[:, 2].astype(np.float64) if dtype == np.float32 else events[:, 2]
                selected = events[(times >= start_us) & (times < end_us)]
                if len(selected):
                    chunks.append(selected)
                # TIME_HIGH can arrive at a chunk end before its TIME_LOW. The
                # previous low bits may temporarily overestimate the clock; use
                # the high-word lower bound until a new low word is consumed.
                clock = (state.time_high << 12) + (0 if state.time_high_updated_since_low else state.time_low)
                if clock - self.shift_us >= end_us + margin:
                    break
        return np.concatenate(chunks) if chunks else np.empty((0, 4), dtype=dtype)
