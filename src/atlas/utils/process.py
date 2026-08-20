"""Process-group lifecycle helpers for subprocess-based workers."""

import os
import signal
import subprocess


def terminate_process_group(process: subprocess.Popen, timeout: float = 15.0) -> None:
    """Terminate a subprocess and its entire process group.

    Sends SIGTERM first and escalates to SIGKILL if the group hasn't
    exited after `timeout` seconds. `process` must have been launched
    with ``start_new_session=True``, so this also reaps any children it
    spawned (e.g. vLLM's own worker processes) rather than orphaning them.

    Parameters
    ----------
    process : subprocess.Popen
        Process launched with ``start_new_session=True``.
    timeout : float, optional
        Seconds to wait after each signal before escalating, by default
        15.0.
    """
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(os.getpgid(process.pid), sig)
        except (ProcessLookupError, PermissionError):
            return
        try:
            process.wait(timeout=timeout)
            return
        except subprocess.TimeoutExpired:
            continue
