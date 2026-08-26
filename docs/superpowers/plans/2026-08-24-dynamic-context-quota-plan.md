# Dynamic context quota implementation plan

Date: 2026-08-24
Design: `docs/superpowers/specs/2026-08-24-dynamic-context-quota-design.md`

## Scope and constraints

Repositories in scope:

- `/data2t/working/PycharmProjects/llama-box`
- `/data2t/working/PycharmProjects/aistack`

The existing `stable-diffusion.cpp` modification in llama-box is unrelated and
must remain untouched. No image rebuild, host deployment, push, or PR is part
of this implementation pass. The llama.cpp repository instructions prohibit
automated commits without explicit approval, so implementation changes and
the design/plan documents remain reviewable in the worktree until separately
authorized.

## 1. Establish the runtime mode boundary

Files:

- `llama-box/engine_param.hpp`
- `llama.cpp/common/common.h`
- `llama.cpp/common/common.cpp`
- `llama.cpp/src/llama-context.cpp`
- `llama-box/httpserver.hpp`

Work:

1. Reuse the existing `common_params::kv_unified` and `--kv-unified`
   argument. Do not add a second flag unless implementation reveals that
   unified KV and dynamic scheduling must be independently controlled.
2. Confirm that target, draft, and MTP contexts inherit the unified setting.
3. In the HTTP server, treat `llama_n_ctx(llm_ctx)` as the total capacity in
   unified mode and keep the current divided capacity in non-unified mode.
4. Keep `--parallel` as the maximum admitted sequence count in both modes.
5. Set the global KV safety limit from the total context in unified mode.

Verification:

- `llama-box --help` accepts `--kv-unified`.
- Startup logs show `kv_unified=true` and `n_ctx_seq` equal to the total
  context for a small test model.
- Non-unified startup still reports the divided per-sequence context.

## 2. Add task admission state and quota helpers

File: `llama-box/httpserver.hpp`

Work:

1. Extend `completions_task` with:
   - an admission marker;
   - the dynamic context limit;
   - the original output limit, normalized so unlimited requests can be
     expanded after another task finishes.
2. Add a reconcile-thread-owned registry of admitted completion tasks keyed by
   task id. Store pointers only while the owning `unique_ptr` task is alive.
3. Add small helpers for:
   - detecting dynamic mode;
   - computing `floor(total_context / active_count)`;
   - checking whether existing task positions fit a candidate quota;
   - calculating remaining output budget from current position, requested
     output, and context limit;
   - registering and releasing a task.
4. Keep all registry mutations on the reconcile thread. HTTP handlers only
   enqueue tasks as they do today.

Verification:

- The helpers have deterministic 1, 2, and 3 active-task results.
- A task release removes its registry entry on every terminal path.
- No pointer in the registry survives destruction of its task.

## 3. Gate completion admission in the reconcile loop

File: `llama-box/httpserver.hpp`

Work:

1. Before selecting a sequence or placing the first prompt batch, calculate
   the candidate quota for the pending completion task.
2. Keep the task queued when the maximum active sequence count is reached.
3. Keep it queued when an admitted task position is above the candidate quota.
4. Keep it queued when the new prompt is above the candidate quota.
5. Reject a prompt above total context with a terminal client error so it does
   not wait forever.
6. Register a task only after it passes admission and receives its sequence id.
7. Refresh existing tasks' future budgets after admission. Do not remove or
   shift their stored KV history because of the new task.
8. Release registry state before destroying a task after normal completion,
   disconnect, prefill failure, or decode failure.
9. Preserve the existing global KV usage check and prompt-cache inactive
   accounting as the final safety layer.
10. Add a debug wait message with task id, active count, candidate quota, and
    the largest active position. Avoid a tight retry loop if the queue is
    otherwise idle.

Verification:

- One active task is admitted with the full total context.
- A second task is admitted only when all existing task positions fit half of
  the total context.
- A second task waits without allocating KV when the first task is above half.
- Completion of the first task wakes the queued task through normal queue
  processing.

## 4. Apply per-task limits to completion paths

File: `llama-box/httpserver.hpp`

Work:

1. Change completion prompt checks in legacy completion and chat completion
   handlers to use total context in dynamic mode. Keep static mode checks
   unchanged.
2. Defer active-count-dependent prompt admission to the reconcile loop.
3. Replace completion uses of the static `llm_slot_ctx_size` with the task's
   dynamic limit after admission, including:
   - initial generation budget;
   - context-shift decisions;
   - decode budget and stopping at the task limit;
   - multimodal prompt paths;
   - speculative MTP output limits.
4. Preserve the existing static limit for embedding and reranking paths.
5. When a task's active count changes, recompute only its remaining generation
   budget. Never truncate its existing prompt or generated KV state.
6. Keep the total KV guard active for batched prefill and decode failures.

Verification:

- A prompt that fits total context but not the current dynamic quota remains
  pending instead of returning a premature 400.
- An explicit `max_tokens` value is capped by the task's remaining context.
- An unlimited request stops at its dynamic context limit when context shift is
  disabled.
- Context shifting stays within the task quota when enabled.
- Qwen3.8 vision and MTP requests pass through the same admission path.

## 5. Expose unambiguous runtime metadata

File: `llama-box/httpserver.hpp`

Work:

1. Keep `meta.n_ctx` as the configured total context.
2. Add `meta.dynamic_context` as a boolean mode marker.
3. Add `meta.n_ctx_single` as the maximum context for one request. In dynamic
   mode it equals the total context; in static mode it equals the existing
   per-slot value.
4. Keep `meta.n_slot` and `meta.n_slot_ctx` backward compatible for static
   clients.

Verification:

- `/v1/models` reports `dynamic_context=true` and `n_ctx_single=ctx-size`
  with unified mode.
- Non-unified metadata remains compatible with current clients.

## 6. Opt Qwen3.8 into the mode in AIStack

Files:

- `/data2t/working/PycharmProjects/aistack/backend/aistack/assets/model-catalog.yaml`
- `/data2t/working/PycharmProjects/aistack/backend/tests/test_qwen38_catalog.py`
- `/data2t/working/PycharmProjects/aistack/backend/aistack/worker/backends/llama_box.py`
- `/data2t/working/PycharmProjects/aistack/backend/tests/test_llama_box_mtp.py`
- `/data2t/working/PycharmProjects/aistack/backend/tests/test_llama_box_embedding.py`

Work:

1. Add `--kv-unified` to the Qwen3.8 llama-box catalog parameters.
2. Update the catalog contract test to require the exact new parameter.
3. Do not append unified mode to encoder defaults or globally force it for
   unrelated models in this pass.
4. Confirm the backend preserves catalog parameters and forwards the flag
   without creating duplicate `--parallel` or `--ctx-size` values.

Verification:

- Existing llama-box parameter tests pass.
- Qwen3.8 command construction contains exactly one `--kv-unified`.
- Embedding and reranker defaults remain unchanged.

## 7. Update playground metadata consumption

Files:

- `/data2t/working/PycharmProjects/aistack/gui/src/pages/playground/components/params-settings.tsx`
- `/data2t/working/PycharmProjects/aistack/gui/tests/playground-model-change.test.mjs`

Work:

1. Read `dynamic_context` and `n_ctx_single` from model metadata.
2. Use `n_ctx_single` as the max-token context display for dynamic models.
3. Retain `n_ctx / n_slot` for static models and existing vLLM metadata.
4. Keep the change limited to context display and default max-token
   initialization; do not change request payload names.

Verification:

- Dynamic metadata no longer displays `n_ctx / n_slot` as the single-request
  maximum.
- Static llama-box and vLLM model behavior remains unchanged.
- Targeted GUI model-switch tests pass.

## 8. Add focused runtime regression coverage

Files:

- `tests/test_dynamic_context.sh` (new, if the existing test harness cannot
  express the concurrent request scenario)
- Existing llama-box test scripts and AIStack tests as applicable

Work:

1. Use a small local GGUF model supplied through a test argument or environment
   variable and a total context of 4096.
2. Start llama-box with `--parallel=4 --kv-unified`.
3. Send one long-lived request and verify it can use the single-request limit.
4. Send a second request while the first is above 2048 and verify that it
   remains pending, not rejected and not allocated into the KV cache.
5. Finish the first request and verify the second completes with the expanded
   available context.
6. Start a non-unified server and verify the old 1024 per-slot behavior.
7. Use bounded timeouts and cleanup traps so a failed test cannot leave a
   server process behind.

If a new test script is not needed, extend an existing server test harness
instead of adding another test file.

## 9. Build and verification sequence

Run read-only checks before editing, then execute the following after each
logical change:

1. `cmake --build build --target llama-box -j$(nproc)` from llama-box.
2. `bash tests/test_mtp_option.sh build/bin/llama-box`.
3. The focused dynamic-context integration test with a small model.
4. From AIStack backend, run:
   `pytest -q tests/test_llama_box_mtp.py tests/test_llama_box_embedding.py tests/test_qwen38_catalog.py`.
5. From AIStack GUI, run the existing targeted playground model-change test
   command from `package.json`.
6. Run the Qwen3.8 startup and request regression only if the required model
   bundle is available locally. Do not claim it passed when only the small
   model test was run.
7. Inspect both worktrees with `git status` and verify only scoped files
   changed. Preserve all unrelated dirty files.

## 10. Delivery boundary

After verification, report:

- changed files and behavior;
- static versus dynamic compatibility results;
- exact test commands and pass/fail output;
- any unavailable model or GPU validation;
- uncommitted changes and the existing stable-diffusion.cpp dirty state.

Do not commit, push, rebuild an image, or deploy to 179 until the user gives
separate explicit authorization for that action.
