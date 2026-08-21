"""Launch and supervise spawned, programmatic vLLM server workers."""

import multiprocessing
import os

from atlas.log import logger
from atlas.schema import WorkerConfig


def build_vllm_args(config: WorkerConfig) -> list[str]:
    """Translate a worker schema into supported vLLM server arguments."""
    args = [
        "--model",
        config.model,
        "--served-model-name",
        config.served_model_name or config.model,
        "--host",
        config.host,
        "--port",
        str(config.port),
        "--tensor-parallel-size",
        str(config.gpu.tensor_parallel_size),
        "--dtype",
        config.dtype,
        "--gpu-memory-utilization",
        str(config.gpu_memory_utilization),
    ]
    optional = (
        ("--max-model-len", config.max_model_len),
        ("--quantization", config.quantization),
        ("--max-num-seqs", config.scheduler.max_num_seqs),
        ("--max-num-batched-tokens", config.scheduler.max_num_batched_tokens),
        (
            "--max-num-scheduled-tokens",
            config.scheduler.max_num_scheduled_tokens,
        ),
        ("--scheduling-policy", config.scheduler.scheduling_policy),
        ("--stream-interval", config.scheduler.stream_interval),
        ("--kv-cache-memory-bytes", config.cache.kv_cache_memory_bytes),
        ("--kv-cache-dtype", config.cache.cache_dtype),
        ("--reasoning-parser", config.api.reasoning_parser),
        ("--tool-call-parser", config.api.tool_call_parser),
    )
    for flag, value in optional:
        if value is not None:
            args += [flag, str(value)]
    # vLLM defaults these to "decide at runtime" and exposes a --no-
    # counterpart for each, so an explicit False has to be sent as
    # --no-<flag>. Emitting nothing would leave vLLM's default in force
    # and silently ignore the request to disable the feature.
    tristate = (
        ("enable-chunked-prefill", config.scheduler.enable_chunked_prefill),
        ("async-scheduling", config.scheduler.async_scheduling),
        ("enable-prefix-caching", config.cache.enable_prefix_caching),
    )
    for name, value in tristate:
        if value is not None:
            args += [f"--{name}" if value else f"--no-{name}"]
    # Unlike the flags above, vLLM defaults this one to False, so
    # omitting it already means "off".
    if config.api.enable_auto_tool_choice:
        args += ["--enable-auto-tool-choice"]
    return args


def _serve_worker(serialized_config: dict[str, object]) -> None:
    """Set CUDA visibility, then import and run vLLM in this child process."""
    config = WorkerConfig.model_validate(serialized_config)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
        str(d) for d in config.gpu.devices
    )

    import uvloop  # noqa: PLC0415
    from vllm.entrypoints.openai.api_server import (  # noqa: PLC0415
        make_arg_parser,
        run_server,
    )
    from vllm.entrypoints.openai.cli_args import (  # noqa: PLC0415
        validate_parsed_serve_args,
    )
    from vllm.entrypoints.serve.utils.api_utils import (  # noqa: PLC0415
        cli_env_setup,
    )
    from vllm.utils.argparse_utils import (  # noqa: PLC0415
        FlexibleArgumentParser,
    )

    cli_env_setup()
    parser = make_arg_parser(
        FlexibleArgumentParser(description="Atlas vLLM server")
    )
    args = parser.parse_args(build_vllm_args(config))
    validate_parsed_serve_args(args)
    uvloop.run(run_server(args))


class Worker:
    """One spawned Python process hosting a programmatic vLLM server."""

    def __init__(self, config: WorkerConfig) -> None:
        """Store one resolved worker configuration."""
        self.config = config
        self._process: multiprocessing.Process | None = None

    @property
    def name(self) -> str:
        """Return the configured display name."""
        return self.config.name

    def start(self) -> None:
        """Start a fresh worker before it imports vLLM or Torch."""
        context = multiprocessing.get_context("spawn")
        self._process = context.Process(
            target=_serve_worker,
            args=(self.config.model_dump(mode="json"),),
            name=f"{self.name}-vllm",
        )
        logger.bind(worker=self.name).info(
            f"launching vLLM server on {self.config.host}:{self.config.port}"
        )
        self._process.start()

    def is_alive(self) -> bool:
        """Return whether the spawned worker process is alive."""
        return self._process is not None and self._process.is_alive()

    def exit_code(self) -> int | None:
        """Return the worker exit code, or None while it is still running."""
        return None if self._process is None else self._process.exitcode

    def terminate(self, timeout: float = 15.0) -> None:
        """Stop the worker, escalating from terminate to kill if required."""
        if self._process is None or not self._process.is_alive():
            return
        self._process.terminate()
        self._process.join(timeout)
        if self._process.is_alive():
            self._process.kill()
            self._process.join(timeout)
