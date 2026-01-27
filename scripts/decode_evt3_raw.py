import argparse
from pathlib import Path

import numpy as np

from ecfm.utils.evt3 import decode_evt3_raw


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Decode Prophesee EVT3 .raw event stream to .npy (x,y,t,p)."
    )
    parser.add_argument("input", type=Path, help="Path to .raw file")
    parser.add_argument("output", type=Path, help="Path to output .npy file")
    parser.add_argument(
        "--endian",
        choices=["little", "big"],
        default="little",
        help="Byte order of EVT3 words (default: little)",
    )
    args = parser.parse_args()

    events, counters, meta, header_lines = decode_evt3_raw(
        args.input, endian=args.endian, require_evt3=True
    )
    np.save(args.output, events)

    print(f"Decoded {events.shape[0]} events -> {args.output}")
    if header_lines:
        print("Header summary:")
        for line in header_lines:
            print(line)
    if any(counters.values()):
        print("Ignored word types:")
        for k, v in counters.items():
            if v:
                print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
