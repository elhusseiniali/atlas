"""Tests for vLLM command construction and process management."""

import os

import pytest

from atlas.schema import GPUConfig, WorkerConfig
from atlas.worker import Worker, build_command


def test_build_command_includes_required_worker_settings() -> None:
    """Build a vLLM command from resolved worker settings."""
    config = WorkerConfig(
        model="example/model",
        served_model_name="example",
        host="127.0.0.1",
        port=8123,
        gpu=GPUConfig(devices=[2, 3]),
        gpu_memory_utilization=0.8,
        max_model_len=4096,
        quantization="fp8",
        extra_args=["--enable-prefix-caching"],
    )

    assert build_command(config) == [
        "vllm",
        "serve",
        "example/model",
        "--served-model-name",
        "example",
        "--host",
        "127.0.0.1",
        "--port",
        "8123",
        "--tensor-parallel-size",
        "2",
        "--dtype",
        "auto",
        "--gpu-memory-utilization",
        "0.8",
        "--max-model-len",
        "4096",
        "--quantization",
        "fp8",
        "--enable-prefix-caching",
    ]


def test_build_command_omits_optional_flags_when_unset() -> None:
    """Avoid passing absent optional values to vLLM."""
    config = WorkerConfig(
        model="example/model",
        port=8000,
        gpu=GPUConfig(devices=[0]),
    )

    command = build_command(config)

    assert "--max-model-len" not in command
    assert "--quantization" not in command
    assert command[4] == "example/model"


def test_worker_start_sets_gpu_visibility_and_spawns_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Launch a worker with only its configured physical GPUs visible."""
    calls: dict[str, object] = {}

    class FakeProcess:
        """Minimal process placeholder."""

    class FakeThread:
        """Thread double that never starts the output pump."""

        def __init__(self, **kwargs) -> None:
            """Record thread construction arguments."""
            calls["thread"] = kwargs

        def start(self) -> None:
            """Avoid starting a background thread in this unit test."""

    monkeypatch.setattr(
        "atlas.worker.subprocess.Popen",
        lambda command, **kwargs: (
            calls.update(command=command, popen=kwargs) or FakeProcess()
        ),
    )
    monkeypatch.setattr("atlas.worker.threading.Thread", FakeThread)
    config = WorkerConfig(
        model="example/model", port=8000, gpu=GPUConfig(devices=[2, 3])
    )
    worker = Worker(config)

    worker.start()

    assert calls["popen"]["env"]["CUDA_VISIBLE_DEVICES"] == "2,3"
    assert calls["popen"]["start_new_session"] is True
    assert calls["command"] == build_command(config)
    assert calls["thread"]["name"] == "example/model-output"


def test_output_pump_forwards_lines_and_throttles_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Forward normal output while emitting only the first rapid progress update."""
    messages: list[str] = []

    class BoundLogger:
        """Logger double that records forwarded output."""

        def info(self, message: str) -> None:
            """Record an output message."""
            messages.append(message)

    class Process:
        """Process double with an output pipe."""

        def __init__(self) -> None:
            """Create a closed writer and readable output stream."""
            read_fd, write_fd = os.pipe()
            os.write(write_fd, b"ready\n25%\r50%\rdone")
            os.close(write_fd)
            self.stdout = os.fdopen(read_fd, "rb", buffering=0)

    worker = Worker(
        WorkerConfig(model="example/model", port=8000, gpu=GPUConfig(devices=[0]))
    )
    worker._process = Process()  # type: ignore[assignment]
    monkeypatch.setattr("atlas.worker.logger.bind", lambda **kwargs: BoundLogger())
    monotonic_values = iter([10.0, 11.0])
    monkeypatch.setattr("atlas.worker.time.monotonic", lambda: next(monotonic_values))

    worker._pump_output()

    assert messages == ["ready", "25%", "done"]


def test_worker_state_and_termination_delegate_to_process_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Expose process state and join the output thread after termination."""
    calls: dict[str, object] = {}

    class Process:
        """Process double with a configurable exit status."""

        def poll(self) -> int | None:
            """Report an active process."""
            return None

    class Thread:
        """Thread double that records join timeouts."""

        def join(self, timeout: float) -> None:
            """Record the requested join timeout."""
            calls["join_timeout"] = timeout

    worker = Worker(
        WorkerConfig(model="example/model", port=8000, gpu=GPUConfig(devices=[0]))
    )
    process = Process()
    worker._process = process  # type: ignore[assignment]
    worker._pump_thread = Thread()  # type: ignore[assignment]
    monkeypatch.setattr(
        "atlas.worker.terminate_process_group",
        lambda received, timeout: calls.update(process=received, timeout=timeout),
    )

    assert worker.is_alive() is True
    assert worker.exit_code() is None
    worker.terminate(timeout=4)

    assert calls == {"process": process, "timeout": 4, "join_timeout": 4}
