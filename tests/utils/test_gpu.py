"""Tests for NVIDIA SMI-backed GPU availability checks."""

import subprocess

import pytest

from atlas.utils.gpu import (
    GPUUnavailableError,
    check_gpu_availability,
    get_gpu_memory_usage,
)

_USED_MIB = 200
_TOTAL_MIB = 2000
_USED_FRACTION = 0.1


def _completed(output: str) -> subprocess.CompletedProcess[str]:
    """Create a successful mocked nvidia-smi result.

    Parameters
    ----------
    output : str
        CSV output reported by nvidia-smi.

    Returns
    -------
    subprocess.CompletedProcess[str]
        Successful command result containing the supplied output.
    """
    return subprocess.CompletedProcess(
        ["nvidia-smi"], 0, stdout=output, stderr=""
    )


def test_get_gpu_memory_usage_parses_requested_devices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Parse GPU memory records from nvidia-smi CSV output."""
    monkeypatch.setattr(
        "atlas.utils.gpu.subprocess.run",
        lambda *args, **kwargs: _completed("0, 100, 1000\n1, 200, 2000\n"),
    )

    usage = get_gpu_memory_usage([1])

    assert usage[1].used_mib == _USED_MIB
    assert usage[1].total_mib == _TOTAL_MIB
    assert usage[1].used_fraction == _USED_FRACTION


def test_missing_requested_gpu_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Report a requested GPU that nvidia-smi did not return."""
    monkeypatch.setattr(
        "atlas.utils.gpu.subprocess.run",
        lambda *args, **kwargs: _completed("0, 100, 1000\n"),
    )

    with pytest.raises(GPUUnavailableError, match="GPU indices not reported"):
        get_gpu_memory_usage([1])


def test_malformed_nvidia_smi_output_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Convert malformed nvidia-smi output into a domain-specific error."""
    monkeypatch.setattr(
        "atlas.utils.gpu.subprocess.run",
        lambda *args, **kwargs: _completed("not,csv\n"),
    )

    with pytest.raises(GPUUnavailableError, match="could not parse"):
        get_gpu_memory_usage()


def test_gpu_usage_equal_to_limit_is_allowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Allow a GPU whose occupancy exactly equals the configured limit."""
    monkeypatch.setattr(
        "atlas.utils.gpu.subprocess.run",
        lambda *args, **kwargs: _completed("0, 500, 1000\n"),
    )

    check_gpu_availability([0], 0.5)


def test_gpu_usage_above_limit_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject a GPU whose occupancy is above the configured limit."""
    monkeypatch.setattr(
        "atlas.utils.gpu.subprocess.run",
        lambda *args, **kwargs: _completed("0, 501, 1000\n"),
    )

    with pytest.raises(GPUUnavailableError, match="exceeding the 50% limit"):
        check_gpu_availability([0], 0.5)
