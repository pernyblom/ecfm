"""Appendable, lossless storage for prepared indexed event-image cutouts."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sqlite3
import zlib

import numpy as np
import torch


class IndexedInputCache:
    """One SQLite file, with a separate connection in each DataLoader process.

    Entries are complete compressed float32 tensors. Autocommit makes each
    insert atomic; WAL allows readers while another worker appends a cutout.
    No RAW events or full rendered intermediates are stored.
    """

    VERSION = 1

    def __init__(self, directory: Path, *, compression_level: int = 1) -> None:
        self.path = Path(directory) / f"indexed_cutouts_v{self.VERSION}.sqlite3"
        self.compression_level = int(compression_level)
        if not 0 <= self.compression_level <= 9:
            raise ValueError("data.instance_cache.compression_level must be between 0 and 9.")
        self._connection = None
        self._pid = None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Initialize before spawning workers, then leave no open parent handle.
        connection = self._connect()
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute(
            "CREATE TABLE IF NOT EXISTS cutouts ("
            "key BLOB PRIMARY KEY, shape TEXT NOT NULL, tensor BLOB NOT NULL) WITHOUT ROWID"
        )
        self.close()

    def _connect(self) -> sqlite3.Connection:
        if self._connection is not None and self._pid != os.getpid():
            self.close()
        if self._connection is None:
            self._connection = sqlite3.connect(self.path, timeout=60.0, isolation_level=None)
            self._connection.execute("PRAGMA synchronous=NORMAL")
            self._pid = os.getpid()
        return self._connection

    def get(self, key: bytes) -> torch.Tensor | None:
        row = self._connect().execute("SELECT shape, tensor FROM cutouts WHERE key = ?", (key,)).fetchone()
        if row is None:
            return None
        shape, blob = row
        # Copy supplies writable tensor memory and isolates callers from the cache.
        array = np.frombuffer(zlib.decompress(blob), dtype="<f4").reshape(json.loads(shape)).copy()
        return torch.from_numpy(array)

    def put(self, key: bytes, tensor: torch.Tensor) -> None:
        array = np.asarray(tensor.numpy(), dtype="<f4", order="C")
        blob = zlib.compress(array.tobytes(), level=self.compression_level)
        self._connect().execute(
            "INSERT OR IGNORE INTO cutouts (key, shape, tensor) VALUES (?, ?, ?)",
            (key, json.dumps(array.shape), blob),
        )

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
        self._connection = None
        self._pid = None

    def __getstate__(self):
        return {**self.__dict__, "_connection": None, "_pid": None}

    def __del__(self):
        if getattr(self, "_connection", None) is not None:
            self.close()
