"""Validates, launches, and supervises a fleet of workers."""

import signal
import time
from types import FrameType

from atlas.config import WorkerConfig
from atlas.log import logger
from atlas.utils.gpu import GPUUnavailableError, check_gpu_availability
from atlas.worker import Worker

_DEFAULT_BASE_PORT = 8000
_POLL_INTERVAL = 2.0


def _resolve_ports(configs: list[WorkerConfig]) -> None:
    """Assign a port to every config that doesn't already have one, in place.

    Parameters
    ----------
    configs : list[atlas.config.WorkerConfig]
        Worker configs to resolve ports for.

    Raises
    ------
    ValueError
        If two configs already have the same explicit port.
    """
    explicit = [c.port for c in configs if c.port is not None]
    if len(explicit) != len(set(explicit)):
        raise ValueError(f"duplicate explicit ports in worker configs: {explicit}")

    taken = set(explicit)
    next_port = _DEFAULT_BASE_PORT
    for config in configs:
        if config.port is not None:
            continue
        while next_port in taken:
            next_port += 1
        config.port = next_port
        taken.add(next_port)


def _check_gpus(configs: list[WorkerConfig]) -> None:
    for config in configs:
        check_gpu_availability(config.gpu.devices, config.gpu.max_used_memory_fraction)


class Orchestrator:
    """Validates, launches, and supervises every worker in a fleet.

    Parameters
    ----------
    configs : list[atlas.config.WorkerConfig]
        Worker configurations to run.
    """

    def __init__(self, configs: list[WorkerConfig]) -> None:
        self._configs = configs
        self._workers: list[Worker] = []
        self._shutdown = False

    def run(self) -> int:
        """Validate, launch, and supervise the fleet until shutdown.

        Returns
        -------
        int
            0 on clean shutdown, 1 if a preflight check or launch failed.
        """
        _resolve_ports(self._configs)
        try:
            _check_gpus(self._configs)
        except GPUUnavailableError as exc:
            logger.error(f"GPU preflight check failed: {exc}")
            return 1

        self._install_signal_handlers()
        if not self._launch_all():
            return 1
        self._supervise()
        return 0

    def _launch_all(self) -> bool:
        for config in self._configs:
            worker = Worker(config)
            try:
                worker.start()
            except OSError as exc:
                logger.error(f"failed to launch worker '{config.name}': {exc}")
                for started in self._workers:
                    started.terminate()
                return False
            self._workers.append(worker)
        return True

    def _install_signal_handlers(self) -> None:
        def _handle(signum: int, _frame: FrameType | None) -> None:
            logger.info(f"received signal {signum}, shutting down...")
            self._shutdown = True

        signal.signal(signal.SIGINT, _handle)
        signal.signal(signal.SIGTERM, _handle)

    def _supervise(self) -> None:
        active = list(self._workers)
        while active and not self._shutdown:
            for worker in list(active):
                if not worker.is_alive():
                    logger.error(
                        f"worker '{worker.name}' exited unexpectedly "
                        f"(code {worker.exit_code()}); leaving other workers running."
                    )
                    active.remove(worker)
            if active:
                time.sleep(_POLL_INTERVAL)

        for worker in self._workers:
            if worker.is_alive():
                logger.info(f"stopping worker '{worker.name}'...")
                worker.terminate()


def run(configs: list[WorkerConfig]) -> int:
    """Validate, launch, and supervise a fleet of workers until shutdown.

    Parameters
    ----------
    configs : list[atlas.config.WorkerConfig]
        Worker configurations to run.

    Returns
    -------
    int
        Process exit code.
    """
    return Orchestrator(configs).run()
