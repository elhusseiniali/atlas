# Atlas

Atlas launches and supervises local `vllm serve` processes from a Python
configuration file. Define one or more workers in `config.py`, then start
them with:

```bash
uv run atlas
```

Atlas writes its output, including each worker's vLLM logs, to `atlas.log`.

## Configuring workers

`config.py` exports a `WORKERS` list. Each `WorkerConfig` becomes one vLLM
server process.

```python
from atlas.config import GPUConfig, WorkerConfig

WORKERS = [
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
```

The main settings are:

- `model`: Hugging Face model ID or local model path.
- `served_model_name`: name clients pass in API requests. Defaults to `model`.
- `gpu.devices`: physical GPU indices assigned to the worker.
- `gpu_memory_utilization`: fraction of each assigned GPU's memory vLLM may
  reserve; defaults to `0.9`.
- `max_model_len`: maximum context length. Omit it to use vLLM's default.
- `extra_args`: additional vLLM command-line flags.

### GPU assignment and preflight check

Atlas sets `CUDA_VISIBLE_DEVICES` for each worker to the GPUs in
`GPUConfig(devices=[...])`. Tensor parallelism defaults to the number of
devices listed. If specified explicitly, `tensor_parallel_size` must equal the
number of devices.

Before starting a worker, Atlas checks the assigned GPUs with `nvidia-smi`.
By default, it refuses to start when a GPU is already more than 50% occupied.
A GPU with some existing memory use at or below that limit produces a warning
but is allowed.

Configure the limit for an individual worker with
`max_used_memory_fraction`:

```python
gpu=GPUConfig(
    devices=[2, 3],
    max_used_memory_fraction=0.75,
)
```

The value must be greater than `0` and no greater than `1`. Setting it to
`1.0` disables this occupancy safeguard, but does not make an already-full GPU
usable by vLLM.

### Ports

An unspecified port is assigned automatically, starting at `8000`. Workers are
processed in `WORKERS` order; automatic ports skip any explicitly assigned
ports. Explicit duplicate ports are rejected before workers are launched.

For example, the following workers listen on ports `8000` and `8001`:

```python
WORKERS = [
    WorkerConfig(model="model-a", gpu=GPUConfig(devices=[0])),
    WorkerConfig(model="model-b", gpu=GPUConfig(devices=[1]), port=8001),
]
```

Set `host` or `port` on a worker when the defaults (`0.0.0.0` and automatic
assignment) are not appropriate.

## Verify a running server

Use the worker's `served_model_name` in OpenAI-compatible requests. For the
example Qwen worker above:

```bash
curl --fail-with-body --silent --show-error \
  http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-27b",
    "prompt": "Reply with exactly: Atlas is working.",
    "max_tokens": 16,
    "temperature": 0
  }'
```

A successful response contains `Atlas is working.` in its generated text.

## Troubleshooting

### Multiple CUDA versions installed

A machine can have several CUDA toolkits installed. `nvidia-smi` reports the
CUDA version supported by the NVIDIA driver, while FlashInfer uses the `nvcc`
compiler found first on `PATH`; the two versions can differ.

For example, this machine reported CUDA 13 in `nvidia-smi`, but `/usr/bin/nvcc`
was CUDA 11.5. FlashInfer requires CUDA 12 or later, so its runtime kernel
compilation failed. Select the intended toolkit before starting Atlas:

```bash
export CUDA_HOME=/usr/local/cuda-13.1
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

nvcc --version
uv run atlas
```

Confirm that `nvcc --version` reports CUDA 12 or newer. Adjust
`/usr/local/cuda-13.1` to the CUDA toolkit you intend to use.
