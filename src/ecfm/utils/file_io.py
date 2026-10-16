"""Bounded retries for temporary file access conflicts (including Windows locks)."""
import errno
import time
import warnings


def retry_file_io(operation, *, description):
    """Retry a restartable operation; propagate missing/corrupt files immediately.

    Callers must reopen files and recreate any partial state on each attempt.
    Eight attempts allow 4.5 seconds for a scanner or another reader to finish.
    """
    for attempt in range(8):
        try:
            return operation()
        except OSError as error:
            transient = (error.errno in {errno.EACCES, errno.EPERM, errno.EBUSY,
                                         errno.EAGAIN, errno.EINTR}
                         or getattr(error, 'winerror', None) in {5, 32, 33})
            if not transient:
                raise
            if attempt == 7:
                if hasattr(error, 'add_note'):
                    error.add_note(f'{description} failed after 8 file access attempts.')
                raise
            if attempt == 0:
                warnings.warn(f'{description}: temporary file access failure ({error}); '
                              'retrying for up to 4.5 seconds.', RuntimeWarning, stacklevel=2)
            time.sleep(min(0.1 * 2**attempt, 1.0))
