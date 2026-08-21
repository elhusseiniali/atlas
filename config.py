"""atlas worker configuration.

Edit this file to describe the model(s) you want to serve, then run
``uv run atlas`` from this directory. Each `atlas.schema.WorkerConfig`
becomes one ``vllm serve`` process on its own GPU set and port.
"""

from atlas.schema import GPUConfig, WorkerConfig

WORKERS = [
    # Qwen3.8-27B is a 27B BF16 multimodal model. Its ~52 GiB of weights do
    # not fit on one 46 GiB A40, so shard it over two GPUs. Start at 32k
    # context; raise max_model_len later if the desired KV-cache capacity fits.
    WorkerConfig(
        model="Qwen/Qwen3.8-27B",
        served_model_name="qwen3.8-27b",
        gpu=GPUConfig(devices=[0, 1]),
        gpu_memory_utilization=0.9,
        max_model_len=32768,
        extra_args=[
            "--reasoning-parser",
            "qwen3",
            "--enable-auto-tool-choice",
            "--tool-call-parser",
            "qwen3_coder",
        ],
    ),
]
