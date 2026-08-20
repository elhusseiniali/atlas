"""GPU VRAM preflight checks, backed by ``nvidia-smi``.

Shells out to ``nvidia-smi`` rather than depending on a CUDA-aware Python
library (``pynvml``/``torch``): it ships with any NVIDIA driver install
(which vLLM already requires), and querying it from the orchestrator process
never initializes a CUDA context on any device.
"""

import subprocess
from collections.abc import Sequence
from dataclasses import dataclass

from atlas.log import logger

_QUERY_FIELDS = "index,memory.used,memory.total"
_NVIDIA_SMI_TIMEOUT = 10.0


class GPUUnavailableError(RuntimeError):
    """Raised when a requested GPU can't be queried or is too heavily used."""


@dataclass(frozen=True)
class GPUMemoryInfo:
    """VRAM usage for a single GPU, as reported by ``nvidia-smi``.

    Parameters
    ----------
    index : int
        GPU index, as reported by ``nvidia-smi``.
    used_mib : int
        VRAM currently in use, in MiB.
    total_mib : int
        Total VRAM on the device, in MiB.
    """

    index: int
    used_mib: int
    total_mib: int

    @property
    def used_fraction(self) -> float:
        """float: Fraction of this GPU's VRAM currently in use, in [0, 1]."""
        return self.used_mib / self.total_mib


def get_gpu_memory_usage(
    devices: Sequence[int] | None = None,
) -> dict[int, GPUMemoryInfo]:
    """Query current VRAM usage for GPUs on this machine.

    Parameters
    ----------
    devices : collections.abc.Sequence[int], optional
        GPU indices that must be present in ``nvidia-smi``'s output, by
        default None, in which case every GPU ``nvidia-smi`` reports is
        returned without checking for specific indices.

    Returns
    -------
    dict[int, atlas.utils.gpu.GPUMemoryInfo]
        Per-GPU memory usage, keyed by GPU index.

    Raises
    ------
    atlas.utils.gpu.GPUUnavailableError
        If ``nvidia-smi`` isn't installed, fails, or doesn't report one of
        the requested `devices`.
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={_QUERY_FIELDS}",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=_NVIDIA_SMI_TIMEOUT,
        )
    except FileNotFoundError as exc:
        raise GPUUnavailableError(
            "nvidia-smi not found; is the NVIDIA driver installed?"
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise GPUUnavailableError(
            f"nvidia-smi exited {exc.returncode}: {exc.stderr.strip()}"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise GPUUnavailableError(
            f"nvidia-smi timed out after {_NVIDIA_SMI_TIMEOUT}s"
        ) from exc

    usage: dict[int, GPUMemoryInfo] = {}
    try:
        for line in result.stdout.strip().splitlines():
            index_str, used_str, total_str = (part.strip() for part in line.split(","))
            index = int(index_str)
            usage[index] = GPUMemoryInfo(
                index=index, used_mib=int(used_str), total_mib=int(total_str)
            )
    except ValueError as exc:
        raise GPUUnavailableError(
            f"could not parse nvidia-smi output: {result.stdout.strip()!r}"
        ) from exc

    if devices is not None:
        missing = set(devices) - usage.keys()
        if missing:
            raise GPUUnavailableError(
                f"GPU indices not reported by nvidia-smi: {sorted(missing)} "
                f"(available: {sorted(usage)})"
            )
    return usage


def check_gpu_availability(devices: Sequence[int], max_used_fraction: float) -> None:
    """Check that `devices` have enough free VRAM to be used.

    Parameters
    ----------
    devices : collections.abc.Sequence[int]
        GPU indices to check.
    max_used_fraction : float
        A device already using more than this fraction of its VRAM fails
        the check; a device in use but at or below it only logs a warning.

    Raises
    ------
    atlas.utils.gpu.GPUUnavailableError
        If any device is already using more than `max_used_fraction` of
        its VRAM (or can't be queried at all).
    """
    usage = get_gpu_memory_usage(devices)
    for index in devices:
        info = usage[index]
        fraction = info.used_fraction
        if fraction > max_used_fraction:
            raise GPUUnavailableError(
                f"GPU {index} is already {fraction:.0%} occupied "
                f"({info.used_mib}/{info.total_mib} MiB), exceeding the "
                f"{max_used_fraction:.0%} limit."
            )
        if fraction > 0:
            logger.warning(
                f"GPU {index} is already in use: {fraction:.0%} occupied "
                f"({info.used_mib}/{info.total_mib} MiB), but within the "
                f"{max_used_fraction:.0%} limit."
            )
