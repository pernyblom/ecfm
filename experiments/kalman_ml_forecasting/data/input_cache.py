"""Appendable, lossless storage for prepared event and RGB image cutouts."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sqlite3
import zlib

import numpy as np
import torch

from ecfm.utils.file_io import is_transient_file_error, retry_file_io


_RETRYABLE_SQLITE_IO_ERRORS = {
    getattr(sqlite3, name, None) for name in (
        "SQLITE_IOERR", "SQLITE_IOERR_ACCESS", "SQLITE_IOERR_READ", "SQLITE_IOERR_WRITE",
        "SQLITE_IOERR_FSYNC", "SQLITE_IOERR_DIR_FSYNC", "SQLITE_IOERR_FSTAT", "SQLITE_IOERR_LOCK",
        "SQLITE_IOERR_RDLOCK", "SQLITE_IOERR_UNLOCK", "SQLITE_IOERR_CLOSE", "SQLITE_IOERR_CHECKRESERVEDLOCK",
        "SQLITE_IOERR_SHMOPEN", "SQLITE_IOERR_SHMSIZE", "SQLITE_IOERR_SHMLOCK", "SQLITE_IOERR_SHMMAP",
        "SQLITE_READONLY_CANTLOCK",
    )
} - {None}


class IndexedInputCache:
    """One SQLite file, with a separate connection in each DataLoader process.

    Entries are complete compressed float32 tensors. Autocommit makes each
    insert atomic; WAL allows readers while another worker appends a cutout.
    Transient access errors restart the operation with a fresh connection.
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
        retry_file_io(lambda: self.path.parent.mkdir(parents=True, exist_ok=True),
                      description=f"Creating input cache directory {self.path.parent}")
        # Initialize before spawning workers, then leave no open parent handle.
        try:
            self._run(self._initialize, description=f"Initializing input cache {self.path}")
        finally:
            self.close()

    @staticmethod
    def _initialize(connection):
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute(
            "CREATE TABLE IF NOT EXISTS cutouts ("
            "key BLOB PRIMARY KEY, shape TEXT NOT NULL, tensor BLOB NOT NULL) WITHOUT ROWID"
        )

    def _is_transient_error(self, error):
        if is_transient_file_error(error):
            return True
        if not isinstance(error, sqlite3.OperationalError):
            return False
        code = getattr(error, "sqlite_errorcode", None)
        if code is not None:
            if (code & 0xff) in {sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED}:
                return True
            if code in _RETRYABLE_SQLITE_IO_ERRORS:
                return True
            if code != sqlite3.SQLITE_CANTOPEN:
                return False
        else:
            # Older Python versions do not expose SQLite error codes.
            message = str(error).lower()
            if message in {"database is locked", "database table is locked", "disk i/o error"}:
                return True
            if message != "unable to open database file":
                return False
        # A scanner may block sqlite3_open itself. Missing parent directories
        # and a directory in place of the database are configuration errors.
        try:
            return self.path.parent.is_dir() and not self.path.is_dir()
        except OSError as stat_error:
            return is_transient_file_error(stat_error)

    def _run(self, operation, *, description):
        def attempt():
            try:
                return operation(self._connect())
            except (OSError, sqlite3.Error):
                self.close()
                raise
        return retry_file_io(attempt, description=description, retry_if=self._is_transient_error)

    def _connect(self) -> sqlite3.Connection:
        if self._connection is not None and self._pid != os.getpid():
            self.close()
        if self._connection is None:
            # A short busy timeout absorbs normal writer contention. Longer
            # access failures use the same bounded backoff as the RAW reader.
            self._connection = sqlite3.connect(self.path, timeout=0.1, isolation_level=None)
            self._connection.execute("PRAGMA synchronous=NORMAL")
            self._pid = os.getpid()
        return self._connection

    def get(self, key: bytes) -> torch.Tensor | None:
        row = self._run(
            lambda connection: connection.execute(
                "SELECT shape, tensor FROM cutouts WHERE key = ?", (key,)).fetchone(),
            description=f"Reading input cache {self.path}",
        )
        if row is None:
            return None
        shape, blob = row
        # Copy supplies writable tensor memory and isolates callers from the cache.
        array = np.frombuffer(zlib.decompress(blob), dtype="<f4").reshape(json.loads(shape)).copy()
        return torch.from_numpy(array)

    def put(self, key: bytes, tensor: torch.Tensor) -> None:
        array = np.asarray(tensor.numpy(), dtype="<f4", order="C")
        blob = zlib.compress(array.tobytes(), level=self.compression_level)
        self._run(
            lambda connection: connection.execute(
                "INSERT OR IGNORE INTO cutouts (key, shape, tensor) VALUES (?, ?, ?)",
                (key, json.dumps(array.shape), blob)),
            description=f"Writing input cache {self.path}",
        )

    def close(self) -> None:
        connection = self._connection
        self._connection = None
        self._pid = None
        if connection is not None:
            connection.close()

    def __getstate__(self):
        return {**self.__dict__, "_connection": None, "_pid": None}

    def __del__(self):
        if getattr(self, "_connection", None) is not None:
            self.close()
