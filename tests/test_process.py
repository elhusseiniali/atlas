"""Tests for process-group termination."""

import signal
import subprocess

import pytest

from atlas.utils.process import terminate_process_group


class _Process:
    """Minimal subprocess double that records waits."""

    pid = 123

    def __init__(self, waits: list[object]) -> None:
        """Store wait outcomes.

        Parameters
        ----------
        waits : list[object]
            Values to return or raise for successive wait calls.
        """
        self.waits = waits
        self.timeouts: list[float] = []

    def wait(self, timeout: float) -> object:
        """Record a timeout and return or raise the next configured outcome."""
        self.timeouts.append(timeout)
        outcome = self.waits.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def test_terminate_process_group_stops_after_sigterm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Send SIGTERM and do not escalate when the group exits."""
    signals: list[tuple[int, signal.Signals]] = []
    monkeypatch.setattr("atlas.utils.process.os.getpgid", lambda pid: 456)
    monkeypatch.setattr(
        "atlas.utils.process.os.killpg", lambda pgid, sig: signals.append((pgid, sig))
    )
    process = _Process([None])

    terminate_process_group(process, timeout=3)

    assert signals == [(456, signal.SIGTERM)]
    assert process.timeouts == [3]


def test_terminate_process_group_escalates_after_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Escalate to SIGKILL only when SIGTERM did not end the group."""
    signals: list[tuple[int, signal.Signals]] = []
    monkeypatch.setattr("atlas.utils.process.os.getpgid", lambda pid: 456)
    monkeypatch.setattr(
        "atlas.utils.process.os.killpg", lambda pgid, sig: signals.append((pgid, sig))
    )
    process = _Process([subprocess.TimeoutExpired("worker", 3), None])

    terminate_process_group(process, timeout=3)

    assert signals == [(456, signal.SIGTERM), (456, signal.SIGKILL)]
    assert process.timeouts == [3, 3]


def test_terminate_process_group_ignores_missing_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return cleanly when the process group already exited."""
    monkeypatch.setattr("atlas.utils.process.os.getpgid", lambda pid: 456)
    monkeypatch.setattr(
        "atlas.utils.process.os.killpg",
        lambda pgid, sig: (_ for _ in ()).throw(ProcessLookupError()),
    )
    process = _Process([None])

    terminate_process_group(process)

    assert process.timeouts == []
