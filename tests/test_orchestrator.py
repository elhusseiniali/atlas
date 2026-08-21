"""Tests for worker port and GPU orchestration checks."""

import pytest

from atlas.orchestrator import Orchestrator, _check_gpus, _resolve_ports
from atlas.schema import GPUConfig, WorkerConfig
from atlas.utils.gpu import GPUUnavailableError

_DEFAULT_PORT = 8000


def _worker(name: str, devices: list[int], port: int | None = None) -> WorkerConfig:
    """Build a minimal worker configuration for orchestration tests.

    Parameters
    ----------
    name : str
        Model and served-model name.
    devices : list[int]
        Physical GPU indices assigned to the worker.
    port : int | None, optional
        Explicit port, if any.

    Returns
    -------
    WorkerConfig
        Minimal valid worker configuration.
    """
    return WorkerConfig(
        model=name,
        served_model_name=name,
        gpu=GPUConfig(devices=devices),
        port=port,
    )


def test_ports_default_to_8000_in_worker_order() -> None:
    """Assign sequential default ports to workers without explicit ports."""
    configs = [_worker("first", [0]), _worker("second", [1])]

    _resolve_ports(configs)

    assert [config.port for config in configs] == [8000, 8001]


def test_automatic_ports_skip_explicit_reservations() -> None:
    """Assign the lowest available ports while preserving explicit ports."""
    configs = [
        _worker("explicit", [0], port=8001),
        _worker("first-auto", [1]),
        _worker("second-auto", [2]),
    ]

    _resolve_ports(configs)

    assert [config.port for config in configs] == [8001, 8000, 8002]


def test_duplicate_explicit_ports_are_rejected() -> None:
    """Reject workers that explicitly claim the same HTTP port."""
    configs = [_worker("first", [0], port=8000), _worker("second", [1], port=8000)]

    with pytest.raises(ValueError, match="duplicate explicit ports"):
        _resolve_ports(configs)


def test_overlapping_worker_gpu_assignments_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject two workers that claim the same physical GPU."""
    checked: list[tuple[list[int], float]] = []
    monkeypatch.setattr(
        "atlas.orchestrator.check_gpu_availability",
        lambda devices, limit: checked.append((list(devices), limit)),
    )

    with pytest.raises(ValueError, match="GPU 1 is assigned to both workers"):
        _check_gpus([_worker("first", [0, 1]), _worker("second", [1, 2])])

    assert checked == []


def test_gpu_preflight_uses_each_workers_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pass configured device lists and thresholds to GPU availability checks."""
    checked: list[tuple[list[int], float]] = []
    monkeypatch.setattr(
        "atlas.orchestrator.check_gpu_availability",
        lambda devices, limit: checked.append((list(devices), limit)),
    )
    configs = [
        _worker("first", [0]),
        WorkerConfig(
            model="second",
            gpu=GPUConfig(devices=[1], max_used_memory_fraction=0.75),
        ),
    ]

    _check_gpus(configs)

    assert checked == [([0], 0.5), ([1], 0.75)]


def test_launch_failure_terminates_previously_started_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Clean up already-started workers when a later launch fails."""
    created: list[object] = []

    class FakeWorker:
        """Worker double that fails only for the second configuration."""

        def __init__(self, config: WorkerConfig) -> None:
            """Store configuration and register the fake worker."""
            self.config = config
            self.terminated = False
            created.append(self)

        def start(self) -> None:
            """Fail the second worker launch."""
            if self.config.name == "second":
                raise OSError("cannot spawn")

        def terminate(self) -> None:
            """Record cleanup."""
            self.terminated = True

    monkeypatch.setattr("atlas.orchestrator.Worker", FakeWorker)
    orchestrator = Orchestrator([_worker("first", [0]), _worker("second", [1])])

    assert orchestrator._launch_all() is False
    assert created[0].terminated is True


def test_supervise_terminates_live_workers_on_shutdown() -> None:
    """Stop every remaining live worker when shutdown has been requested."""

    class FakeWorker:
        """Worker double representing an active process."""

        name = "example"

        def __init__(self) -> None:
            """Initialize termination state."""
            self.terminated = False

        def is_alive(self) -> bool:
            """Report the worker as active until termination."""
            return not self.terminated

        def terminate(self) -> None:
            """Record termination."""
            self.terminated = True

    worker = FakeWorker()
    orchestrator = Orchestrator([])
    orchestrator._workers = [worker]  # type: ignore[assignment]
    orchestrator._shutdown = True

    orchestrator._supervise()

    assert worker.terminated is True


def test_run_stops_before_launch_when_gpu_preflight_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return a failure status without installing handlers or launching workers."""
    orchestrator = Orchestrator([_worker("example", [0])])
    monkeypatch.setattr(
        "atlas.orchestrator._check_gpus",
        lambda configs: (_ for _ in ()).throw(GPUUnavailableError("busy")),
    )
    monkeypatch.setattr(
        orchestrator,
        "_install_signal_handlers",
        lambda: pytest.fail("handlers should not be installed"),
    )

    assert orchestrator.run() == 1


def test_run_resolves_ports_then_launches_and_supervises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run the normal orchestration sequence after successful preflight."""
    orchestrator = Orchestrator([_worker("example", [0])])
    calls: list[str] = []
    monkeypatch.setattr(
        "atlas.orchestrator._check_gpus", lambda configs: calls.append("gpu")
    )
    monkeypatch.setattr(
        orchestrator, "_install_signal_handlers", lambda: calls.append("signals")
    )
    monkeypatch.setattr(
        orchestrator, "_launch_all", lambda: calls.append("launch") or True
    )
    monkeypatch.setattr(orchestrator, "_supervise", lambda: calls.append("supervise"))

    assert orchestrator.run() == 0
    assert orchestrator._configs[0].port == _DEFAULT_PORT
    assert calls == ["gpu", "signals", "launch", "supervise"]
