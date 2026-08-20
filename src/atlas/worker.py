"""Builds and owns a single ``vllm serve`` subprocess."""

import os
import re
import subprocess
import threading
import time

from atlas.config import WorkerConfig
from atlas.log import logger
from atlas.utils.process import terminate_process_group

# Progress bars redraw with "\r"; regular log lines end with "\n".
_TERMINATOR_RE = re.compile(rb"[\r\n]")
_READ_CHUNK_SIZE = 8192
_PROGRESS_LOG_INTERVAL = 5.0


def build_command(config: WorkerConfig) -> list[str]:
    """Translate a worker config into a ``vllm serve`` command line.

    Parameters
    ----------
    config : atlas.config.WorkerConfig
        Worker configuration to translate.

    Returns
    -------
    list[str]
        Command line, suitable for `subprocess.Popen`.
    """
    args = [
        "vllm",
        "serve",
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
    if config.max_model_len is not None:
        args += ["--max-model-len", str(config.max_model_len)]
    if config.quantization is not None:
        args += ["--quantization", config.quantization]
    args += config.extra_args
    return args


class Worker:
    """One ``vllm serve`` subprocess, on its own GPU set and port.

    Parameters
    ----------
    config : atlas.config.WorkerConfig
        Configuration for this worker. `config.port` must already be
        resolved to a concrete port (see `atlas.orchestrator`).
    """

    def __init__(self, config: WorkerConfig) -> None:
        self.config = config
        self._process: subprocess.Popen | None = None
        self._pump_thread: threading.Thread | None = None

    @property
    def name(self) -> str:
        """str: Display name for this worker, used in logs."""
        return self.config.name

    def start(self) -> None:
        """Launch the ``vllm serve`` subprocess for this worker."""
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(d) for d in self.config.gpu.devices)

        command = build_command(self.config)
        logger.bind(worker=self.name).info(f"launching: {' '.join(command)}")
        self._process = subprocess.Popen(
            command,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        self._pump_thread = threading.Thread(
            target=self._pump_output, daemon=True, name=f"{self.name}-output"
        )
        self._pump_thread.start()

    def _pump_output(self) -> None:
        """Re-emit the subprocess's output through loguru as it arrives.

        Splits on carriage returns as well as newlines. Progress bars (tqdm,
        used by both the Hugging Face downloader and vLLM's weight loader)
        redraw in place with ``\\r`` and emit no newline until the bar
        finishes, so iterating the stream by line would buffer an entire
        multi-hour download into a single line and display nothing until it
        completed. Carriage-return updates are throttled to one every
        `_PROGRESS_LOG_INTERVAL` seconds, since logging every redraw would
        write millions of lines to the log file.
        """
        assert self._process is not None
        assert self._process.stdout is not None

        bound_logger = logger.bind(worker=self.name)
        fd = self._process.stdout.fileno()
        buffer = b""
        last_progress = 0.0

        while True:
            chunk = os.read(fd, _READ_CHUNK_SIZE)
            if not chunk:
                break
            buffer += chunk
            while (match := _TERMINATOR_RE.search(buffer)) is not None:
                text = buffer[: match.start()].decode("utf-8", errors="replace")
                is_progress = match.group() == b"\r"
                buffer = buffer[match.end() :]

                text = text.rstrip()
                if not text:
                    continue
                if is_progress:
                    now = time.monotonic()
                    if now - last_progress < _PROGRESS_LOG_INTERVAL:
                        continue
                    last_progress = now
                bound_logger.info(text)

        if (text := buffer.decode("utf-8", errors="replace").rstrip()) != "":
            bound_logger.info(text)

    def is_alive(self) -> bool:
        """bool: Whether the subprocess is still running."""
        return self._process is not None and self._process.poll() is None

    def exit_code(self) -> int | None:
        """int, optional: The subprocess's exit code, or None if still running."""
        return None if self._process is None else self._process.poll()

    def terminate(self, timeout: float = 15.0) -> None:
        """Terminate this worker's process group.

        Parameters
        ----------
        timeout : float, optional
            Seconds to wait after each signal before escalating, by
            default 15.0.
        """
        if self._process is None:
            return
        terminate_process_group(self._process, timeout=timeout)
        if self._pump_thread is not None:
            self._pump_thread.join(timeout=timeout)
