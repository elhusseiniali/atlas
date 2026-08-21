"""Tests for configuration-file loading and command-line startup."""

from pathlib import Path

import pytest

from atlas import cli


def test_load_config_returns_workers_from_python_file(tmp_path: Path) -> None:
    """Load the module-level worker list from a user configuration file."""
    config_path = tmp_path / "workers.py"
    config_path.write_text(
        "from atlas.schema import GPUConfig, WorkerConfig\n"
        "WORKERS = [WorkerConfig("
        "model='example/model', gpu=GPUConfig(devices=[0])"
        ")]\n"
    )

    workers = cli.load_config(config_path)

    assert len(workers) == 1
    assert workers[0].model == "example/model"


def test_load_config_rejects_missing_file(tmp_path: Path) -> None:
    """Raise a clear error when the requested configuration file is absent."""
    with pytest.raises(FileNotFoundError, match="config file not found"):
        cli.load_config(tmp_path / "missing.py")


def test_load_config_rejects_missing_workers(tmp_path: Path) -> None:
    """Require a module-level WORKERS definition."""
    config_path = tmp_path / "workers.py"
    config_path.write_text("VALUE = 1\n")

    with pytest.raises(AttributeError, match="no module-level 'WORKERS' list"):
        cli.load_config(config_path)


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        ("WORKERS = ()\n", "must be a list"),
        ("WORKERS = [object()]\n", "must contain only WorkerConfig"),
    ],
)
def test_load_config_rejects_invalid_worker_definitions(
    tmp_path: Path,
    contents: str,
    message: str,
) -> None:
    """Reject non-list or non-WorkerConfig ``WORKERS`` definitions."""
    config_path = tmp_path / "workers.py"
    config_path.write_text(contents)

    with pytest.raises(TypeError, match=message):
        cli.load_config(config_path)


def test_main_passes_configured_workers_to_orchestrator(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Configure logging and pass loaded workers to the orchestrator."""
    worker = object()
    configured: dict[str, object] = {}
    monkeypatch.setattr(cli, "load_config", lambda path: [worker])
    monkeypatch.setattr(
        cli,
        "configure_logging",
        lambda **kwargs: configured.update(kwargs),
    )
    monkeypatch.setattr(
        cli, "run_workers", lambda workers: 7 if workers == [worker] else 1
    )
    monkeypatch.setattr(
        "sys.argv",
        ["atlas", "--config", str(tmp_path / "config.py")],
    )

    with pytest.raises(SystemExit, match="7"):
        cli.main()

    assert configured == {"level": "INFO"}
