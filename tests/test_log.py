"""Tests for shared Loguru configuration."""

from pathlib import Path

import pytest

from atlas import log


def test_configure_logging_adds_stderr_and_file_sinks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Configure the expected stderr and file destinations."""
    calls: list[tuple[object, dict[str, object]]] = []
    monkeypatch.setattr(log.logger, "remove", lambda: None)
    monkeypatch.setattr(log.logger, "configure", lambda **kwargs: None)
    monkeypatch.setattr(
        log.logger,
        "add",
        lambda destination, **kwargs: calls.append((destination, kwargs)),
    )
    destination = tmp_path / "atlas.log"

    log.configure_logging(level="DEBUG", log_file=destination)

    assert calls[0][1]["level"] == "DEBUG"
    assert calls[0][1]["colorize"] is True
    assert calls[1][0] == destination
    assert calls[1][1]["colorize"] is False


def test_configure_logging_can_disable_file_sink(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Leave only stderr configured when the file destination is None."""
    calls: list[tuple[object, dict[str, object]]] = []
    monkeypatch.setattr(log.logger, "remove", lambda: None)
    monkeypatch.setattr(log.logger, "configure", lambda **kwargs: None)
    monkeypatch.setattr(
        log.logger,
        "add",
        lambda destination, **kwargs: calls.append((destination, kwargs)),
    )

    log.configure_logging(log_file=None)

    assert len(calls) == 1
