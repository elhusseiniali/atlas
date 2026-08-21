"""Pydantic schemas for worker configuration.

The root-level ``config.py`` that a user edits imports :class:`WorkerConfig`
(and :class:`GPUConfig`) from here to build a ``WORKERS`` list. Validation
happens at construction time, before anything is launched.
"""

from pydantic import BaseModel, Field, model_validator


class GPUConfig(BaseModel):
    """GPU placement and safety settings for one worker.

    Parameters
    ----------
    devices : list[int]
        GPU indices (as reported by ``nvidia-smi``) this worker's
        ``vllm serve`` process should see, via ``CUDA_VISIBLE_DEVICES``.
    tensor_parallel_size : int, optional
        Number of GPUs to shard the model across, by default None, in
        which case it's derived from ``len(devices)``. If given
        explicitly, it must equal ``len(devices)``.
    max_used_memory_fraction : float, optional
        Upper bound on how much of a device's VRAM may already be in use
        before launch, as a fraction in ``(0, 1]``, by default 0.5. A
        device above this fraction fails the preflight GPU check; a
        device in use but below it only logs a warning.
    """

    devices: list[int] = Field(min_length=1)
    tensor_parallel_size: int | None = None
    max_used_memory_fraction: float = Field(default=0.5, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def _resolve_tensor_parallel_size(self) -> "GPUConfig":
        if self.tensor_parallel_size is None:
            self.tensor_parallel_size = len(self.devices)
        elif self.tensor_parallel_size != len(self.devices):
            raise ValueError(
                f"tensor_parallel_size ({self.tensor_parallel_size}) must equal "
                f"len(devices) ({len(self.devices)})"
            )
        return self


class WorkerConfig(BaseModel):
    """Configuration for a single ``vllm serve`` worker.

    Parameters
    ----------
    model : str
        Model repo id (e.g. a Hugging Face path) or local path to serve.
    gpu : atlas.schema.GPUConfig
        GPU placement and safety settings for this worker.
    served_model_name : str, optional
        Name clients use to request this model, by default None, in
        which case it defaults to `model`.
    host : str, optional
        Host to bind the server to, by default "0.0.0.0".
    port : int, optional
        Port to bind the server to, by default None, in which case one
        is auto-assigned (starting at 8000, incrementing per worker).
    dtype : str, optional
        Model dtype passed to vLLM, by default "auto".
    max_model_len : int, optional
        Maximum sequence length, by default None (vLLM's own default).
    gpu_memory_utilization : float, optional
        Fraction of GPU memory vLLM may reserve, by default 0.9.
    quantization : str, optional
        Quantization method passed to vLLM, by default None.
    extra_args : list[str], optional
        Additional raw CLI arguments appended to the ``vllm serve``
        command, as an escape hatch for flags this schema doesn't model
        directly, by default an empty list.
    """

    model: str
    gpu: GPUConfig
    served_model_name: str | None = None
    host: str = "0.0.0.0"
    port: int | None = None
    dtype: str = "auto"
    max_model_len: int | None = None
    gpu_memory_utilization: float = Field(default=0.9, gt=0.0, le=1.0)
    quantization: str | None = None
    extra_args: list[str] = Field(default_factory=list)

    @property
    def name(self) -> str:
        """str: Display name for this worker, used in logs."""
        return self.served_model_name or self.model
