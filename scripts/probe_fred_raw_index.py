"""Read-only investigation of existing FRED/OpenEB v2 RAW indexes.

This is a comparison probe, not a production reader. It intentionally uses the
repository decoder (including its float32 timestamps) as both candidate and
sequential reference. It creates only a JSON report, never event/image files.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from ecfm.utils.evt3 import _Evt3DecoderState, _decode_evt3_words, read_raw_header


DTYPE = np.dtype([('t', '<i8'), ('offset', '<u8'), ('count', '<u4')])
# OpenEB generates bytes in native struct order (offset, timestamp, count),
# then serializes fields in timestamp, offset, count order on these Linux files.
_magic = np.random.RandomState(0x6D76).randint(0, 2**32, 20, dtype=np.uint32).astype(np.uint8).tobytes()
MAGIC = _magic[8:16] + _magic[:8] + _magic[16:20]


def read_index(raw: Path):
    sidecar = raw.with_suffix(raw.suffix + '.tmp_index')
    _, offset, meta = read_raw_header(sidecar)
    _, raw_offset, raw_meta = read_raw_header(raw)
    payload = sidecar.stat().st_size - offset
    if meta.get('index_version') != '2.0' or payload % 20 or payload < 40:
        raise ValueError('Unsupported/incomplete index layout')
    if int(meta['size']) != raw.stat().st_size:
        raise ValueError('RAW size differs from index header')
    if any(meta.get(k) != v for k, v in raw_meta.items()):
        raise ValueError('RAW header differs from index header')
    records = np.fromfile(sidecar, dtype=DTYPE, offset=offset)
    if records[-1].tobytes() != MAGIC:
        raise ValueError('Missing/unrecognized completion marker')
    records = records[:-1]
    valid = records[records['t'] >= 0]
    if not len(valid) or np.any(np.diff(valid['t']) < 0):
        raise ValueError('Invalid timestamp ordering')
    offsets = records['offset'].astype(np.int64)
    if (np.any(np.diff(offsets) < 0) or np.any(offsets < raw_offset)
            or np.any(offsets >= raw.stat().st_size)
            or np.any((offsets - raw_offset) % 2)):
        raise ValueError('Invalid RAW offsets')
    return records, int(meta['ts_shift_us']), int(meta['bookmark_period_us']), raw_offset


def state_at(timestamp_us: int):
    high = timestamp_us >> 12
    return _Evt3DecoderState(
        time_low=timestamp_us & 0xFFF, time_high=high,
        time_high_raw=high & 0xFFF, time_high_base=high & ~0xFFF,
        last_time_high_raw=high & 0xFFF, last_time_low=timestamp_us & 0xFFF,
    )


def extract(raw, records, shift, period, start, end, *, guard_us=4000):
    started = time.perf_counter()
    slot = max(0, start - guard_us) // period
    if slot >= len(records):
        raise ValueError('Requested start outside index')
    while slot < len(records) and records[slot]['t'] < 0:
        slot += 1
    if slot == len(records) or int(records[slot]['t']) > start:
        raise ValueError('No preceding valid bookmark; needs start-of-file fallback')
    bookmark = records[slot]
    state = state_at(int(bookmark['t']) + shift)
    chunks, counters, consumed = [], None, 0
    with raw.open('rb') as stream:
        stream.seek(int(bookmark['offset']))
        while True:
            data = stream.read(64 * 1024)
            if not data:
                break
            consumed += len(data)
            events, counters, state = _decode_evt3_words(
                np.frombuffer(data, dtype='<u2'), state=state, counters=counters)
            times = events[:, 2].astype(np.float64) - shift
            chunks.append(events[(times >= start) & (times < end)])
            # Decoder time also progresses during windows with no CD events.
            if (state.time_high << 12 | state.time_low) - shift >= end + 32:
                break
    events = np.concatenate(chunks) if chunks else np.empty((0, 4), np.float32)
    return events, dict(seconds=time.perf_counter() - started, bytes_read=consumed,
                       offset=int(bookmark['offset']), bookmark_us=int(bookmark['t']),
                       preroll_us=start-int(bookmark['t']))


def probe(raw: Path, guard_us: int):
    records, shift, period, offset = read_index(raw)
    last = int(records[-1]['t'])
    anchors = [1_000_000, 10_000_000, 17_000_000, 60_000_000, 110_000_000]
    # Explicitly straddle an absolute EVT3 24-bit wrap in shifted coordinates.
    wrap = ((shift // (1 << 24) + 1) * (1 << 24)) - shift
    anchors.extend([wrap + 16000, wrap + 410000])
    rng = np.random.default_rng(42)
    anchors.extend(int(t) for t in rng.integers(500000, last - 500000, size=5))
    windows = sorted(set((end-width, end) for end in anchors
                         for width in (33333, 400000) if width < end < last))
    references = [[] for _ in windows]
    state, counters, consumed = None, None, 0
    started = time.perf_counter()
    with raw.open('rb') as stream:
        stream.seek(offset)
        while True:
            data = stream.read(1024 * 1024)
            if not data:
                break
            consumed += len(data)
            events, counters, state = _decode_evt3_words(
                np.frombuffer(data, dtype='<u2'), state=state, counters=counters)
            times = events[:, 2].astype(np.float64) - shift
            for parts, (start, end) in zip(references, windows):
                selected = events[(times >= start) & (times < end)]
                if len(selected):
                    parts.append(selected)
            if (state.time_high << 12 | state.time_low) - shift >= max(e for _, e in windows) + 32:
                break
    reference_seconds = time.perf_counter() - started
    results = []
    for parts, (start, end) in zip(references, windows):
        reference = np.concatenate(parts) if parts else np.empty((0, 4), np.float32)
        candidate, stats = extract(raw, records, shift, period, start, end, guard_us=guard_us)
        same = bool(np.array_equal(candidate, reference))
        results.append(dict(start_us=start, end_us=end, reference_events=len(reference),
                            indexed_events=len(candidate), exact_match=same, **stats))
    return dict(sequence=raw.parent.parent.name, guard_us=guard_us,
                reference_seconds=reference_seconds, reference_bytes=consumed,
                matches=sum(r['exact_match'] for r in results), windows=results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('datasets/FRED'))
    parser.add_argument('--sequences', nargs='*', default=['0', '8', '66'])
    parser.add_argument('--guard-us', type=int, default=4000)
    parser.add_argument('--output', type=Path, default=Path('outputs/fred_index_investigation/results.json'))
    args = parser.parse_args()
    audit, index_bytes = [], 0
    for folder in sorted(args.root.iterdir()):
        if not folder.name.isdigit() or int(folder.name) >= 900:
            continue
        raw = folder / 'Event/events.raw'
        if not raw.exists():
            continue
        try:
            records, shift, period, _ = read_index(raw)
            size = raw.with_suffix('.raw.tmp_index').stat().st_size
            index_bytes += size
            audit.append(dict(sequence=folder.name, valid=True, bookmarks=len(records),
                              index_bytes=size, shift_us=shift, period_us=period))
        except Exception as error:
            audit.append(dict(sequence=folder.name, valid=False, error=str(error)))
    report = dict(audit=audit, total_index_bytes=index_bytes, probes=[])
    print(json.dumps(dict(audit_valid=sum(a['valid'] for a in audit), audit_total=len(audit),
                          total_index_MiB=index_bytes/2**20,
                          failures=[a for a in audit if not a['valid']])), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for name in args.sequences:
        result = probe(args.root / name / 'Event/events.raw', args.guard_us)
        report['probes'].append(result)
        print(json.dumps(dict(sequence=name, matches=result['matches'],
                              windows=len(result['windows']),
                              reference_seconds=result['reference_seconds'])), flush=True)
        args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    if not args.sequences:
        args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
