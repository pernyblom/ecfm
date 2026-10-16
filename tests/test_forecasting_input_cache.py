"""Fault injection for temporary access conflicts in the cutout cache."""
from contextlib import closing
import sqlite3

import pytest
import torch

from ecfm.utils import file_io
from experiments.kalman_ml_forecasting.data.input_cache import IndexedInputCache

if not hasattr(sqlite3, 'SQLITE_IOERR'):
    pytest.skip('SQLite fault injection needs the error-code API from Python 3.11+', allow_module_level=True)


@pytest.fixture
def backoffs(monkeypatch):
    delays = []
    monkeypatch.setattr(file_io.time, 'sleep', delays.append)
    return delays


def sqlite_error(code):
    error = sqlite3.OperationalError('simulated file access failure')
    error.sqlite_errorcode = code
    return error


@pytest.mark.parametrize('stage', ['open', 'PRAGMA', 'CREATE', 'SELECT', 'FETCH', 'INSERT', 'AFTER_INSERT', 'CLOSE'])
def test_cache_reopens_after_access_failure(tmp_path, monkeypatch, backoffs, stage):
    original_connect = sqlite3.connect
    failed = []
    connections = []

    def fail_once(connection=None):
        failed.append(connection)
        raise sqlite_error(sqlite3.SQLITE_IOERR_READ)

    class Connection:
        def __init__(self, connection):
            self.connection = connection
            self.closed = False

        def execute(self, statement, *args):
            if not failed and statement.startswith(stage):
                fail_once(self)
            if not failed and stage == 'CLOSE' and statement.startswith('SELECT'):
                fail_once(self)
            cursor = self.connection.execute(statement, *args)
            if not failed and stage == 'AFTER_INSERT' and statement.startswith('INSERT'):
                fail_once(self)  # commit succeeded, but the caller saw an error
            if not failed and stage == 'FETCH' and statement.startswith('SELECT'):
                class FailedFetch:
                    def fetchone(inner):
                        cursor.close()
                        fail_once(self)
                return FailedFetch()
            return cursor

        def close(self):
            self.closed = True
            self.connection.close()
            if stage == 'CLOSE' and self in failed:
                raise sqlite_error(sqlite3.SQLITE_IOERR_CLOSE)

    def connect(*args, **kwargs):
        if not failed and stage == 'open':
            fail_once()
        connection = Connection(original_connect(*args, **kwargs))
        connections.append(connection)
        return connection

    monkeypatch.setattr(sqlite3, 'connect', connect)
    expected = torch.tensor([[[-1.25, 0, .1234567, 1.0e20]]], dtype=torch.float32)
    with pytest.warns(RuntimeWarning, match='temporary file access failure'):
        cache = IndexedInputCache(tmp_path)
        cache.put(b'cutout', expected)
        torch.testing.assert_close(cache.get(b'cutout'), expected, rtol=0, atol=0)
    assert backoffs == [.1]
    assert all(connection is None or connection.closed for connection in failed)
    assert len(connections) >= 2
    with closing(original_connect(cache.path)) as connection:
        assert connection.execute('SELECT COUNT(*) FROM cutouts').fetchone()[0] == 1
    cache.close()


@pytest.mark.parametrize('code', [
    sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED_SHAREDCACHE, sqlite3.SQLITE_CANTOPEN,
    sqlite3.SQLITE_IOERR_READ, sqlite3.SQLITE_IOERR_WRITE, sqlite3.SQLITE_IOERR_SHMOPEN,
    sqlite3.SQLITE_READONLY_CANTLOCK,
])
def test_cache_recovers_from_sqlite_access_errors(tmp_path, monkeypatch, backoffs, code):
    cache = IndexedInputCache(tmp_path)
    expected = torch.ones(3, 4, 4)
    cache.put(b'cutout', expected)
    cache.close()
    original_connect = sqlite3.connect
    attempts = []

    def connect(*args, **kwargs):
        attempts.append(1)
        if len(attempts) <= 2:
            raise sqlite_error(code)
        return original_connect(*args, **kwargs)

    monkeypatch.setattr(sqlite3, 'connect', connect)
    with pytest.warns(RuntimeWarning, match='Reading input cache'):
        torch.testing.assert_close(cache.get(b'cutout'), expected)
    assert len(attempts) == 3 and backoffs == [.1, .2]
    cache.close()


@pytest.mark.parametrize('error', [
    PermissionError(13, 'scanner lock'), sqlite_error(sqlite3.SQLITE_BUSY),
])
def test_cache_persistent_lock_propagates_after_bounded_retries(tmp_path, monkeypatch, backoffs, error):
    cache = IndexedInputCache(tmp_path)
    attempts = []

    def fail(*args, **kwargs):
        attempts.append(1)
        raise error

    monkeypatch.setattr(sqlite3, 'connect', fail)
    with pytest.warns(RuntimeWarning), pytest.raises(type(error)) as caught:
        cache.get(b'cutout')
    assert caught.value is error
    assert len(attempts) == 8 and sum(backoffs) == pytest.approx(4.5)
    assert any(str(cache.path) in note for note in caught.value.__notes__)


@pytest.mark.parametrize('error', [
    sqlite_error(sqlite3.SQLITE_CORRUPT), sqlite_error(sqlite3.SQLITE_NOTADB),
    sqlite_error(sqlite3.SQLITE_FULL), sqlite_error(sqlite3.SQLITE_READONLY),
    sqlite_error(sqlite3.SQLITE_ERROR), sqlite_error(sqlite3.SQLITE_IOERR_SHORT_READ),
    FileNotFoundError(2, 'missing cache'),
])
def test_cache_does_not_retry_permanent_errors(tmp_path, monkeypatch, backoffs, error):
    cache = IndexedInputCache(tmp_path)
    attempts = []

    def fail(*args, **kwargs):
        attempts.append(1)
        raise error

    monkeypatch.setattr(sqlite3, 'connect', fail)
    with pytest.raises(type(error)) as caught:
        cache.get(b'cutout')
    assert caught.value is error
    assert len(attempts) == 1 and backoffs == []


def test_cache_recovers_from_real_sqlite_writer_lock(tmp_path, monkeypatch):
    cache = IndexedInputCache(tmp_path)
    with closing(sqlite3.connect(cache.path)) as writer:
        writer.execute('BEGIN IMMEDIATE')
        backoffs = []

        def release_lock(delay):
            backoffs.append(delay)
            writer.rollback()

        monkeypatch.setattr(file_io.time, 'sleep', release_lock)
        with pytest.warns(RuntimeWarning, match='Writing input cache'):
            cache.put(b'cutout', torch.ones(3, 4, 4))
    torch.testing.assert_close(cache.get(b'cutout'), torch.ones(3, 4, 4))
    assert backoffs == [.1]
    cache.close()


def test_cache_missing_parent_fails_without_retries(tmp_path, backoffs):
    cache = IndexedInputCache(tmp_path)
    cache.path = tmp_path / 'missing' / 'cache.sqlite3'
    with pytest.raises(sqlite3.OperationalError):
        cache.get(b'cutout')
    assert backoffs == []
