"""Tests for programmatic vLLM worker configuration and lifecycle."""

import pytest
from pydantic import ValidationError

from atlas.schema import (
    APIConfig,
    CacheConfig,
    GPUConfig,
    SchedulerConfig,
    WorkerConfig,
)
from atlas.worker import Worker, build_vllm_args

_TENSOR_PARALLEL_SIZE = 2
_MAX_MODEL_LEN = 4096


def test_build_vllm_args_uses_typed_serving_settings() -> None:
    """Translate supported Atlas settings without raw CLI fragments."""
    config = WorkerConfig(
        model="example/model",
        served_model_name="example",
        port=8000,
        gpu=GPUConfig(devices=[0, 1]),
        scheduler=SchedulerConfig(
            max_num_seqs=64,
            max_num_batched_tokens=8192,
            enable_chunked_prefill=True,
        ),
        cache=CacheConfig(enable_prefix_caching=True),
        api=APIConfig(
            reasoning_parser="qwen3",
            enable_auto_tool_choice=True,
            tool_call_parser="qwen3_coder",
        ),
    )

    assert build_vllm_args(config) == [
        "--model",
        "example/model",
        "--served-model-name",
        "example",
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--tensor-parallel-size",
        "2",
        "--dtype",
        "auto",
        "--gpu-memory-utilization",
        "0.9",
        "--max-num-seqs",
        "64",
        "--max-num-batched-tokens",
        "8192",
        "--reasoning-parser",
        "qwen3",
        "--tool-call-parser",
        "qwen3_coder",
        "--enable-chunked-prefill",
        "--enable-prefix-caching",
        "--enable-auto-tool-choice",
    ]


def test_explicitly_disabled_flags_are_sent_as_no_flags() -> None:
    """Send ``--no-`` forms so vLLM's defaults are actually overridden."""
    config = WorkerConfig(
        model="example/model",
        port=8000,
        gpu=GPUConfig(devices=[0]),
        scheduler=SchedulerConfig(
            enable_chunked_prefill=False, async_scheduling=False
        ),
        cache=CacheConfig(enable_prefix_caching=False),
    )

    args = build_vllm_args(config)

    assert "--no-enable-chunked-prefill" in args
    assert "--no-async-scheduling" in args
    assert "--no-enable-prefix-caching" in args
    assert "--enable-chunked-prefill" not in args
    assert "--enable-prefix-caching" not in args


def test_unset_flags_defer_to_vllm() -> None:
    """Emit neither form when a three-state flag is left unset."""
    config = WorkerConfig(
        model="example/model", port=8000, gpu=GPUConfig(devices=[0])
    )

    args = build_vllm_args(config)

    for name in (
        "enable-chunked-prefill",
        "async-scheduling",
        "enable-prefix-caching",
    ):
        assert f"--{name}" not in args
        assert f"--no-{name}" not in args


def test_auto_tool_choice_requires_a_tool_call_parser() -> None:
    """Reject the unusable combination while it is still a config error."""
    with pytest.raises(ValidationError, match="requires tool_call_parser"):
        APIConfig(enable_auto_tool_choice=True)


def test_auto_tool_choice_is_accepted_with_a_parser() -> None:
    """Allow automatic tool choice once a parser is configured."""
    config = APIConfig(
        enable_auto_tool_choice=True, tool_call_parser="qwen3_coder"
    )

    assert config.tool_call_parser == "qwen3_coder"


def test_build_vllm_args_parses_into_vllms_engine_model_field() -> None:
    """Check the built arguments against vLLM's own server parser.

    vLLM's server parser exposes the model twice: as the optional
    positional ``model_tag``, and as ``--model``. Only ``--model``
    reaches ``ModelConfig`` and selects the weights that are loaded;
    ``model_tag`` is read solely by the ``vllm serve`` CLI wrapper,
    which copies it into ``model``. Atlas calls ``run_server`` directly
    and so bypasses that wrapper, meaning a bare positional would leave
    ``args.model`` at vLLM's default checkpoint while the server still
    advertised the configured ``served_model_name``.

    Comparing `atlas.worker.build_vllm_args` output against a literal
    argument list cannot catch that, since both sides would agree. This
    test parses the arguments instead, and is skipped when vLLM isn't
    installed (as in the GPU-free CI environment).
    """
    pytest.importorskip("vllm")

    from vllm.entrypoints.openai.api_server import (  # noqa: PLC0415
        make_arg_parser,
    )
    from vllm.entrypoints.openai.cli_args import (  # noqa: PLC0415
        validate_parsed_serve_args,
    )
    from vllm.utils.argparse_utils import (  # noqa: PLC0415
        FlexibleArgumentParser,
    )

    config = WorkerConfig(
        model="example/model",
        served_model_name="example",
        port=8000,
        gpu=GPUConfig(devices=[0, 1]),
        max_model_len=_MAX_MODEL_LEN,
    )

    parser = make_arg_parser(FlexibleArgumentParser(description="test"))
    args = parser.parse_args(build_vllm_args(config))
    validate_parsed_serve_args(args)

    assert args.model == "example/model"
    assert args.model_tag is None
    assert args.served_model_name == ["example"]
    assert args.tensor_parallel_size == _TENSOR_PARALLEL_SIZE
    assert args.max_model_len == _MAX_MODEL_LEN


def test_worker_spawns_process_without_importing_vllm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pass a serialized schema to a fresh spawned Python process."""
    calls: dict[str, object] = {}

    class Process:
        """Minimal spawned-process double."""

        exitcode = None

        def __init__(self, **kwargs: object) -> None:
            calls["process"] = kwargs

        def start(self) -> None:
            calls["started"] = True

        def is_alive(self) -> bool:
            return True

        def terminate(self) -> None:
            calls["terminated"] = True

        def join(self, timeout: float) -> None:
            calls["timeout"] = timeout

        def kill(self) -> None:
            calls["killed"] = True

    Context = type("Context", (), {"Process": Process})

    monkeypatch.setattr(
        "atlas.worker.multiprocessing.get_context", lambda method: Context()
    )
    worker = Worker(
        WorkerConfig(
            model="example/model", port=8000, gpu=GPUConfig(devices=[2])
        )
    )

    worker.start()

    assert calls["started"] is True
    assert calls["process"]["args"][0]["gpu"]["devices"] == [2]
    assert worker.is_alive() is True
