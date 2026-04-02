# claude-code-vllm
Explore vllm using claude code

This repo is for learning vLLM's design — understanding its architecture, core abstractions, and how it achieves high-throughput LLM inference through techniques like PagedAttention, continuous batching, and tensor parallelism.

---

## vLLM Data / Request Flow

### Overview

```
User Input
    ↓
[InputProcessor]       tokenize prompt, parse sampling params
    ↓
[Scheduler]            batch requests, allocate KV cache blocks
    ↓
[ModelExecutor]        dispatch batch to GPU workers
    ↓
[GPU Worker / Model]   forward pass, paged attention, sample tokens
    ↓
[Scheduler update]     record generated tokens, free finished requests
    ↓
[OutputProcessor]      detokenize, build response
    ↓
API Response
```

---

### Significant Components

#### 1. Entry Points (`vllm/entrypoints/`)
- **`LLM`** — high-level Python API (`llm.generate(...)`)
- **OpenAI API** — HTTP server with `/v1/chat/completions`, `/v1/embeddings`, etc.
- **`AsyncLLM`** — async variant for serving frameworks

#### 2. InputProcessor
- Tokenizes the prompt into `token_ids`
- Parses `SamplingParams` (temperature, top_k, top_p, stop strings, etc.)
- Produces an `EngineCoreRequest` and hands it to the engine

#### 3. LLMEngine + EngineCore (`vllm/v1/engine/`)
- **`LLMEngine`** — public-facing wrapper; calls `step()` in a loop
- **`EngineCore`** — the main execution loop:
  1. Call `Scheduler.schedule()` → get the next batch
  2. Call `ModelExecutor.execute_model()` → run the model
  3. Call `Scheduler.update_from_output()` → advance request states
  4. Return finished outputs

#### 4. Scheduler (`vllm/v1/core/sched/scheduler.py`)
- Maintains three queues: **waiting → running → finished**
- Implements **continuous batching**: each iteration it dynamically picks which requests to include, so the GPU stays busy even as requests finish at different times
- Allocates physical KV cache blocks to each request and builds a block table
- Returns a `SchedulerOutput` describing the exact batch for this step

#### 5. KV Cache Manager (`vllm/v1/core/kv_cache_*.py`)
- Implements **PagedAttention**: instead of pre-allocating one giant contiguous buffer per sequence, memory is split into fixed-size blocks (e.g. 16 tokens each)
- A `BlockPool` manages free/used blocks; the block table maps logical token positions → physical block IDs
- Enables **prefix caching**: blocks whose token content hash matches an existing block are reused without recomputation, saving memory and compute for shared prefixes (e.g. system prompts)

#### 6. ModelExecutor + Workers (`vllm/v1/executor/`, `vllm/v1/worker/`)
- **`ModelExecutor`** — dispatches the batch to one or more workers (via multiprocessing or Ray for multi-GPU)
- **`GPUWorker`** — owns a single GPU; loads the model, manages the paged KV cache tensors
- **`GPUModelRunner`** — prepares the flattened `GPUInputBatch`, builds `AttentionMetadata` (block tables, sequence lengths, positions), runs the model forward pass

#### 7. Attention Backends (`vllm/v1/attention/backends/`)
Pluggable kernels that consume the block table to perform paged attention:
- **FlashAttention** — default for H100/A100
- **FlashInfer** — optimized decode-phase attention
- **Flex Attention** — torch.compile-based
- **CPU** — fallback for CPU inference

Each backend reads K/V values from physical blocks according to the block table, computes attention, and writes newly generated K/V back into the cache.

#### 8. Sampler (`vllm/v1/sample/`)
- Takes raw logits `[batch, vocab_size]` from the model
- Applies logits processors: temperature scaling, top-k/top-p filtering, repetition penalty, etc.
- Samples the next token ID(s) from the resulting distribution
- Optionally computes log-probabilities

#### 9. OutputProcessor
- Detokenizes `output_token_ids` back to text
- Checks stop conditions (EOS token, stop strings, max tokens)
- Assembles the final `RequestOutput` with `text`, `finish_reason`, token counts, and optional logprobs

#### 10. Distributed Execution (`vllm/distributed/`)
- **Tensor Parallelism (TP)**: splits attention heads and MLP weights across GPUs; an `AllReduce` synchronizes after each layer
- **Pipeline Parallelism (PP)**: splits transformer layers across GPUs
- **Data Parallelism (DP)**: multiple independent engine replicas
- Workers communicate via `collective_rpc()` / `torch.distributed`

#### 11. Speculative Decoding (`vllm/v1/spec_decode/`)
- A small draft model proposes several tokens ahead
- The large target model verifies the whole draft in a single forward pass
- A rejection sampler accepts or rolls back tokens, yielding lower latency at the same output quality

---

### Request Lifecycle

```
WAITING      queued, no KV blocks allocated
    ↓
RUNNING      scheduled; prefill (all prompt tokens at once) then
             decode (one new token per step) until stop condition
    ↓
FINISHED     EOS / max_tokens / stop string / abort / error
             → KV blocks freed back to pool
```

---

### Why It's Fast

| Technique | What it solves |
|---|---|
| Continuous batching | GPU idles when requests finish at different times |
| PagedAttention | KV cache fragmentation and inability to share prefixes |
| Prefix caching | Redundant compute for shared system prompts |
| Pipelined async execution | CPU scheduling overlaps with GPU compute |
| Tensor / pipeline parallelism | Single-GPU memory limits for large models |
| Speculative decoding | Memory-bandwidth bottleneck in autoregressive decode |
