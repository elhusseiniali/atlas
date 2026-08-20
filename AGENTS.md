# AGENTS.md

## Project overview

Atlas is a configuration-driven launcher for one or more local `vllm serve`
workers. Users define workers in the root-level `config.py`; Atlas validates,
launches, logs, and supervises the resulting vLLM subprocesses.

## Repository map

- `config.py`: user-facing worker configuration. Keep it concise, readable,
  and directly runnable.
- `src/atlas/config.py`: Pydantic configuration schemas and validation.
- `src/atlas/orchestrator.py`: port resolution, GPU preflight checks, worker
  launch order, and lifecycle supervision.
- `src/atlas/worker.py`: vLLM command construction and subprocess ownership.
- `src/atlas/utils/gpu.py`: GPU memory checks implemented through
  `nvidia-smi`.
- `README.md`: operator-facing installation, configuration, and
  troubleshooting documentation.
- `atlas.log`: runtime log output. Do not commit it.

## Code style and documentation

- Use NumPy-style docstrings for public modules, classes, functions, and
  methods.
- Document parameters, return values, raised exceptions, and relevant side
  effects.
- Include input and output types in docstrings when they are available and
  useful, even when Python type annotations are present.
- Keep docstrings accurate when changing validation, configuration defaults,
  or runtime behavior.
- Follow the repository's Ruff configuration. Use `uv run ruff check .` and
  `uv run ruff format --check .` for relevant changes.

## Configuration conventions

- Each entry in `WORKERS` creates one `vllm serve` process.
- `GPUConfig.devices` contains physical GPU indices, which Atlas passes to the
  worker through `CUDA_VISIBLE_DEVICES`.
- Tensor parallelism defaults to the number of listed devices. If
  `tensor_parallel_size` is explicit, it must equal `len(devices)`.
- Keep model-specific vLLM flags in `extra_args` unless the option belongs in
  the shared `WorkerConfig` schema.
- Preserve explicit port assignments. Unspecified ports are assigned from
  `8000`, in `WORKERS` order, while skipping explicitly reserved ports.
- Do not silently change a model, GPU placement, context length,
  quantization, GPU memory utilization, or exposed port. These are
  operationally significant choices and require an explicit user request.

## GPU and CUDA safety

- Before launch, Atlas refuses a worker when an assigned GPU exceeds
  `GPUConfig.max_used_memory_fraction`, which defaults to `0.5`.
- GPUs with nonzero use at or below the limit generate a warning but do not
  block launch.
- Do not weaken the preflight threshold solely to make a launch proceed.
  Explain the memory trade-off and make the smallest requested change.
- A machine may have multiple CUDA toolkits installed. `nvidia-smi` reports
  the CUDA capability of the driver, while FlashInfer JIT compilation uses
  the `nvcc` found on `PATH`.
- When investigating FlashInfer compilation errors, check `nvcc --version`,
  `CUDA_HOME`, and `PATH`. Do not change system CUDA alternatives, terminate
  GPU processes, or remove model caches unless the user explicitly asks.

## Development workflow

- Use `uv run ...` for project commands.
- Prefer focused unit tests for schema validation, port resolution, command
  construction, GPU checks, and process lifecycle changes. Mock
  subprocesses and `nvidia-smi`; ordinary tests should not require real GPUs
  or downloaded models.
- Run the relevant checks after edits. At minimum, run Ruff for Python
  changes; run affected tests when they exist.
- Do not commit virtual environments, logs, model caches, Hugging Face
  checkpoints, or other generated runtime artifacts.
- Keep `README.md` examples and operational documentation synchronized with
  public schema or behavior changes.

## Runtime diagnosis and verification

- Read `atlas.log` first when diagnosing a failed launch. It contains the
  vLLM subprocess output under the worker name.
- A process launch is not proof that a model is ready. Confirm that the API
  server completed startup, then query `/v1/models` or send a small
  OpenAI-compatible request using the configured `served_model_name`.
- Treat a successful response as confirmation that model loading, engine
  initialization, and request handling all completed.
- Keep diagnostic commands read-only unless the user requested an operational
  change.

## Change boundaries

- Make the smallest change that satisfies the requested behavior.
- Preserve unrelated working-tree changes.
- Explain any change that affects live workers, GPU allocation, network
  exposure, model downloads, or CUDA toolchain selection before applying it.
