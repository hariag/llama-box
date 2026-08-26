# Dynamic total-context allocation for llama-box

Date: 2026-08-24
Status: Draft for review

## Problem

llama-box currently derives a fixed per-slot context from the configured total
context and the maximum parallel sequence count:

```cpp
llm_slot_ctx_size = llm_ctx_size / llm_params.n_parallel;
```

The value is then used for prompt validation, generation budgets, context
shifting, embedding limits, and model metadata. With `--ctx-size=65536`
and `--parallel=4`, one request is therefore limited to about 16384 tokens
even when the other three sequences are idle.

The underlying llama.cpp code already provides `--kv-unified`. In that mode it
uses one KV buffer for all sequences and does not derive `n_ctx_seq` by
dividing the context by `n_seq_max`. The server still needs admission control
and per-request limits to provide predictable behavior when requests overlap.

## Goals

1. In dynamic mode, one active completion request can use the full configured
   context.
2. With `N` active completion requests, each request receives a context quota
   of `floor(total_context / N)` for new admission and future generation.
3. A new request waits when an existing request already occupies more than the
   quota that would result from admitting it.
4. Existing request history is never truncated merely because a new request
   arrived. Existing requests may receive a smaller future generation budget,
   but their current KV state remains intact.
5. When a request finishes, remaining requests can expand their future budget
   and queued requests can be admitted.
6. Static, non-unified operation remains compatible with current behavior.

## Non-goals

- Changing tensor parallelism, tensor split, GPU placement, or worker
  scheduling.
- Resizing a llama context or reallocating GPU memory while the server runs.
- Changing the semantics of embeddings, reranking, or image generation.
- Adding a request-level context size API in this change.

## Approaches considered

### A. Enable unified KV only

Pass `--kv-unified` and remove the server-side fixed slot limit. This is small
and lets the KV allocator share unused space, but a long request can consume
almost the entire cache and leave no predictable capacity for a second request.
It does not satisfy the requested `ctx-size / active_requests` contract.

### B. Unified KV plus dynamic admission quotas (recommended)

Use `--kv-unified` for the physical cache and add a scheduler-owned admission
layer in llama-box. The scheduler tracks admitted completion tasks, computes
the quota for the candidate active count, and keeps a request in the existing
task queue until its prompt and the current active states fit that quota.

This provides the requested behavior without changing the llama.cpp memory
API. It also gives a safe interpretation for a request that arrives after an
earlier request has already grown beyond half of the total context: the later
request waits instead of forcing a context shift or overcommitting the cache.

### C. Recreate or resize contexts as concurrency changes

Create a one-sequence context for a single request and rebuild it when another
request arrives. This could provide exact per-request sizes but would require
KV state migration, causes long pauses, and risks GPU memory fragmentation. It
is not suitable for a serving process.

## Design

### Mode selection and physical KV capacity

Dynamic scheduling is enabled when `--kv-unified` is present. Without it,
llama-box keeps the existing static per-slot behavior.

In unified mode:

- `llm_ctx_size` is the total configured KV context.
- `llm_kv_cache_limit` remains a global safety limit derived from the total
  context.
- The server must not use `llm_ctx_size / n_parallel` as the completion
  context limit.
- `--parallel` remains the maximum number of admitted sequences, not a
  physical context split.

The current Qwen3.8 catalog entry will opt into unified mode. Other model
entries will not be changed automatically until their runtime compatibility
has been checked.

### Admission state

The reconcile thread is the owner of dynamic state, so no new cross-thread
locking is required. Each completion task receives state equivalent to:

- whether it has been admitted;
- its assigned sequence id;
- its current context limit;
- its original requested output limit, so a later expansion can restore
  remaining budget.

The server keeps an admitted-task registry keyed by task id. A task enters the
registry only when its sequence is selected and its prompt is ready to prefill.
It leaves the registry on normal completion, client disconnect, prefill error,
or decode error.

### Admission algorithm

For a pending completion task:

1. Let `active` be the number of admitted completion tasks.
2. If `active >= parallel`, keep the task queued.
3. Set `candidate = active + 1` and
   `quota = floor(total_context / candidate)`.
4. If any admitted task has current position greater than `quota`, keep the
   new task queued. This is the selected no-forced-shrink behavior.
5. If the new prompt is larger than `quota`, keep it queued. If it is larger
   than the total context, return the existing invalid-request error rather
   than wait forever.
6. Otherwise admit the task, assign its sequence, and set its context limit
   to `quota`.

When the active count changes, the scheduler refreshes the future budget of
admitted tasks. It never removes already stored KV positions. The remaining
budget is bounded by both the request's original `max_tokens` and the current
context limit minus the task's current position. An unlimited request uses the
same remaining-context bound.

The existing global KV usage check remains in place as a second defense. Idle
prompt-cache entries continue to be accounted for using the current inactive
cache accounting and may be evicted by the existing cache path.

### Prompt validation, context shift, and queueing

In dynamic mode, HTTP handlers validate a prompt against the total context and
defer active-count-dependent validation to the reconcile thread. A prompt that
fits the total but not the current dynamic quota waits in the normal task
queue; it is not rejected with a 4xx response.

If context shifting is enabled, shifting is performed within the task's
current quota. If it is disabled, generation stops when the task reaches that
quota. The existing total-KV guard remains authoritative for batch decode and
prefill failures.

Waiting is observable through a debug log containing task id, active count,
candidate quota, and the largest active position. The HTTP request remains
open, matching the current synchronous task processing behavior.

### Metadata and frontend behavior

The model metadata will continue to expose `n_ctx` as the configured total.
Dynamic mode will additionally expose `dynamic_context: true` and
`n_ctx_single`, whose value is the maximum context for one request. The static
`n_ctx / n_slot` calculation must not be used for a dynamic model.

The playground context display will use the maximum single-request context for
dynamic models and retain the existing division for static models. Embedding
and reranking displays remain unchanged because those models do not opt into
dynamic completion scheduling.

### Error handling and compatibility

- A task that cannot be admitted is requeued without consuming KV memory.
- A task larger than the total context returns a clear client error.
- Any admission state must be removed on every terminal path to avoid a stale
  active count blocking future requests.
- Non-unified models retain the current static `n_ctx / n_parallel` behavior.
- The Qwen3.8 MTP and multimodal paths use the same task quota and must not
  bypass admission checks.

## Validation plan

Validation will use a small local model and a small context so the behavior is
deterministic and does not require the production Qwen3.8 weights.

1. Build llama-box and verify the binary accepts `--kv-unified`.
2. Start with a total context of 4096 and `--parallel=4`:
   - one request can use a single-request limit of 4096;
   - two admitted requests receive 2048 each;
   - a second request waits if the first is already above 2048;
   - after the first request finishes, the queued request is admitted and can
     use the expanded limit.
3. Verify that non-unified mode still reports and enforces the old static
   per-slot limit.
4. Run existing llama-box build and server checks, plus the existing AIStack
   llama-box parameter tests after adding the Qwen3.8 opt-in.
5. Test Qwen3.8-specific MTP and multimodal startup paths with the rebuilt
   binary before any image rebuild or host deployment.

No production container restart, image rebuild, push, or PR is part of this
design approval step.
