"""Pydantic schemas for worker configuration.

The root-level ``config.py`` that a user edits imports :class:`WorkerConfig`
(and :class:`GPUConfig`) from here to build a ``WORKERS`` list. Validation
happens at construction time, before anything is launched.
"""

from pydantic import BaseModel, Field, model_validator


class SchedulerConfig(BaseModel):
    """vLLM scheduler settings for one worker.

    Parameters
    ----------
    max_num_seqs : int, optional
        Maximum number of sequences vLLM may process concurrently, by
        default None, meaning vLLM's own default applies.
    max_num_batched_tokens : int, optional
        Maximum tokens vLLM may include in one scheduled batch, by
        default None, meaning vLLM's own default applies.
    max_num_scheduled_tokens : int, optional
        Maximum tokens vLLM may schedule in one iteration, by default
        None, meaning vLLM's own default applies.
    enable_chunked_prefill : bool, optional
        Whether to split large prefills across scheduler steps, by
        default None, meaning vLLM's own default applies. See the note
        on three-state flags below.
    scheduling_policy : str, optional
        vLLM scheduling policy to use, by default None, meaning vLLM's
        own default applies.
    async_scheduling : bool, optional
        Whether to overlap scheduling with model execution, by default
        None, meaning vLLM's own default applies.
    stream_interval : int, optional
        Number of generation steps between streamed response updates, by
        default None, meaning vLLM's own default applies.

    Notes
    -----
    `enable_chunked_prefill` and `async_scheduling` are three-state
    rather than plain booleans, because vLLM defaults them to "decide at
    runtime" rather than to off. None leaves that decision to vLLM;
    False disables the feature explicitly. A plain ``bool`` could not
    express the difference, so writing False would be indistinguishable
    from omitting the field and would silently leave vLLM's default in
    force.
    """

    max_num_seqs: int | None = Field(default=None, gt=0)
    max_num_batched_tokens: int | None = Field(default=None, gt=0)
    max_num_scheduled_tokens: int | None = Field(default=None, gt=0)
    enable_chunked_prefill: bool | None = None
    scheduling_policy: str | None = None
    async_scheduling: bool | None = None
    stream_interval: int | None = Field(default=None, gt=0)


class CacheConfig(BaseModel):
    """vLLM KV-cache settings for one worker.

    Parameters
    ----------
    enable_prefix_caching : bool, optional
        Whether to reuse KV cache across shared prompt prefixes, by
        default None, meaning vLLM's own default applies. This is
        three-state for the reason given in
        `atlas.schema.SchedulerConfig`.
    kv_cache_memory_bytes : int, optional
        Fixed number of bytes allocated to vLLM's KV cache, by default
        None, meaning vLLM calculates the cache size from available GPU
        memory.
    cache_dtype : str, optional
        Data type to use for the KV cache, by default None, meaning
        vLLM's own default applies.
    """

    enable_prefix_caching: bool | None = None
    kv_cache_memory_bytes: int | None = Field(default=None, gt=0)
    cache_dtype: str | None = None


class APIConfig(BaseModel):
    """OpenAI-compatible vLLM endpoint settings for one worker.

    Parameters
    ----------
    reasoning_parser : str, optional
        Parser vLLM uses to split reasoning content out of responses, by
        default None.
    enable_auto_tool_choice : bool, optional
        Whether to let the model choose tools automatically, by default
        False, which matches vLLM's own default. Requires
        `tool_call_parser`.
    tool_call_parser : str, optional
        Parser vLLM uses to extract tool calls from responses, by
        default None.

    Raises
    ------
    ValueError
        If `enable_auto_tool_choice` is set without a
        `tool_call_parser`. vLLM enforces this too, but only once the
        worker has been spawned, which would surface as an opaque
        nonzero exit code after other workers had already started
        loading weights.
    """

    reasoning_parser: str | None = None
    enable_auto_tool_choice: bool = False
    tool_call_parser: str | None = None

    @model_validator(mode="after")
    def _require_tool_call_parser(self) -> "APIConfig":
        if self.enable_auto_tool_choice and self.tool_call_parser is None:
            raise ValueError(
                "enable_auto_tool_choice requires tool_call_parser"
            )
        return self


class GPUConfig(BaseModel):
    """GPU placement and safety settings for one worker.

    Parameters
    ----------
    devices : list[int]
        GPU indices (as reported by ``nvidia-smi``) this worker's
        vLLM server process should see, via ``CUDA_VISIBLE_DEVICES``.
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
                f"tensor_parallel_size ({self.tensor_parallel_size}) must "
                "equal "
                f"len(devices) ({len(self.devices)})"
            )
        return self


class WorkerConfig(BaseModel):
    """Configuration for a single vLLM server worker.

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
    """

    model: str
    gpu: GPUConfig
    served_model_name: str | None = None
    host: str = "0.0.0.0"
    port: int | None = Field(default=None, ge=1, le=65535)
    dtype: str = "auto"
    max_model_len: int | None = None
    gpu_memory_utilization: float = Field(default=0.9, gt=0.0, le=1.0)
    quantization: str | None = None
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    api: APIConfig = Field(default_factory=APIConfig)

    @property
    def name(self) -> str:
        """str: Display name for this worker, used in logs."""
        return self.served_model_name or self.model
