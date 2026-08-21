"""Command-line entry point: load the user's config.py and run its workers."""

import argparse
import importlib.util
import sys
from pathlib import Path

from atlas.log import configure_logging, logger
from atlas.orchestrator import run as run_workers
from atlas.schema import WorkerConfig

_DEFAULT_CONFIG_PATH = Path("config.py")


def load_config(path: Path) -> list[WorkerConfig]:
    """Load a user config file and return its ``WORKERS`` list.

    Parameters
    ----------
    path : pathlib.Path
        Path to a Python file defining a module-level ``WORKERS`` list of
        `atlas.schema.WorkerConfig`.

    Returns
    -------
    list[atlas.schema.WorkerConfig]
        The loaded workers.

    Raises
    ------
    FileNotFoundError
        If `path` doesn't exist.
    AttributeError
        If the loaded module has no ``WORKERS`` attribute.
    TypeError
        If ``WORKERS`` is not a list of `atlas.schema.WorkerConfig` instances.
    """
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")

    spec = importlib.util.spec_from_file_location("atlas_user_config", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load config file: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    workers = getattr(module, "WORKERS", None)
    if workers is None:
        raise AttributeError(f"{path} has no module-level 'WORKERS' list")
    if not isinstance(workers, list):
        raise TypeError(f"{path} 'WORKERS' must be a list of WorkerConfig")
    if not all(isinstance(worker, WorkerConfig) for worker in workers):
        raise TypeError(f"{path} 'WORKERS' must contain only WorkerConfig")
    return workers


def main() -> None:
    """Entry point for the ``atlas`` console script."""
    parser = argparse.ArgumentParser(
        description="Launch local vLLM servers from a config file."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_DEFAULT_CONFIG_PATH,
        help=f"path to the config file (default: {_DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="minimum log level to emit (default: INFO)",
    )
    args = parser.parse_args()

    configure_logging(level=args.log_level)
    try:
        workers = load_config(args.config)
    except (FileNotFoundError, AttributeError, ImportError, TypeError) as exc:
        logger.error(str(exc))
        sys.exit(1)

    sys.exit(run_workers(workers))


if __name__ == "__main__":
    main()
