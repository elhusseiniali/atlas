"""Tests for public worker configuration validation."""

import pytest
from pydantic import ValidationError

from atlas.schema import GPUConfig, WorkerConfig

_TWO_GPU_COUNT = 2


def test_tensor_parallel_size_defaults_to_number_of_devices() -> None:
    """Derive tensor parallel size from configured device count."""
    config = GPUConfig(devices=[0, 2])

    assert config.tensor_parallel_size == _TWO_GPU_COUNT


def test_matching_explicit_tensor_parallel_size_is_accepted() -> None:
    """Allow an explicit tensor parallel size that matches device count."""
    config = GPUConfig(devices=[0, 1], tensor_parallel_size=_TWO_GPU_COUNT)

    assert config.tensor_parallel_size == _TWO_GPU_COUNT


def test_mismatched_tensor_parallel_size_is_rejected() -> None:
    """Reject a tensor parallel size that cannot match the GPU assignment."""
    with pytest.raises(ValidationError, match="must equal len\\(devices\\)"):
        GPUConfig(devices=[0, 1], tensor_parallel_size=1)


@pytest.mark.parametrize("value", [0, -0.1, 1.1])
def test_invalid_memory_fraction_is_rejected(value: float) -> None:
    """Reject occupancy thresholds outside the supported range."""
    with pytest.raises(ValidationError):
        GPUConfig(devices=[0], max_used_memory_fraction=value)


@pytest.mark.parametrize("value", [0, -0.1, 1.1])
def test_invalid_gpu_memory_utilization_is_rejected(value: float) -> None:
    """Reject vLLM memory utilization values outside the supported range."""
    with pytest.raises(ValidationError):
        WorkerConfig(
            model="example/model",
            gpu=GPUConfig(devices=[0]),
            gpu_memory_utilization=value,
        )


@pytest.mark.parametrize("port", [0, -1, 65536])
def test_invalid_port_is_rejected(port: int) -> None:
    """Reject ports outside the TCP port range."""
    with pytest.raises(ValidationError):
        WorkerConfig(
            model="example/model",
            gpu=GPUConfig(devices=[0]),
            port=port,
        )


def test_worker_name_prefers_served_model_name() -> None:
    """Use the request-facing model name for worker display names."""
    named = WorkerConfig(
        model="example/model",
        served_model_name="example",
        gpu=GPUConfig(devices=[0]),
    )
    unnamed = WorkerConfig(model="example/model", gpu=GPUConfig(devices=[0]))

    assert named.name == "example"
    assert unnamed.name == "example/model"
