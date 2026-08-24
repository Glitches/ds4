# Qwen 3.6 27B support in ds4

## STATUS — READ FIRST

**This branch does NOT support running the real Qwen 3.6 27B model end-to-end.**

The real `Qwen/Qwen3.6-27B` (GGUF arch `qwen35`, see
[huggingface.co/Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) and
[ggml-org/Qwen3.6-27B-GGUF](https://huggingface.co/ggml-org/Qwen3.6-27B-GGUF))
is a **hybrid linear/full-attention multimodal model**:

- 64 layers in a 3:1 pattern — 75% `linear_attention` (Gated DeltaNet, recurrent
  state, **no per-position KV cache**) and 25% `full_attention` (GQA)
- `hidden_size` 5120, `num_attention_heads` 24, `num_key_value_heads` 4,
  `head_dim` 256, `intermediate_size` 17408, `vocab_size` 248320
- `rope_theta` 1e7, `partial_rotary_factor` 0.25 (only 64 of 256 dims rotate)
- vision encoder + projector (multimodal)

The Metal kernels committed in this branch (`kernel_qwen_store_kv`,
`kernel_qwen_attention_gqa` in `metal/qwen_gqa.metal`) implement **standard GQA
with a per-position KV cache and full rotary**. They can neither express the
Gated-DeltaNet linear-attention layers nor `head_dim 256` nor partial rotary.

**What this branch DOES provide:**
- A correct `DS4_SHAPE_QWEN36_27B` shape table with the real qwen35 dimensions
- A `qwen35` loader branch in `config_validate_model()` so a Qwen 3.6 27B GGUF
  is recognised (instead of failing with `unsupported DeepSeek4 shape`)
- A GQA kernel + graph-forward prototype that is valid for a *standard* dense
  GQA transformer, kept as scaffolding for the full-attention layers
- A macOS smoke-test gate (`tests/qwen_metal_smoke.sh`)

**What is still required to run the real model** (tracked below under
"Remaining work") is a new attention backend: Gated-DeltaNet linear-attention
kernels, full-attention kernels extended to `head_dim 256` + partial rotary,
per-layer-type dispatch, and a recurrent state allocator. This is substantial
new GPU work, not a parameter fix.



## Overview

`qwen36_27b.patch` adds registration of the [Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B)
model with the ds4 Metal backend. It is a standard `git diff` and is applied
with `git apply`.

The patch introduces a small, model-agnostic configuration registry
(`struct ds4_model_config`) that is intentionally separate from the
DeepSeek/GLM-specific `ds4_shape` table. Additional architectures can be
registered the same way without touching the inference core.

## Files

| File                         | Purpose                                              |
|------------------------------|------------------------------------------------------|
| `qwen36_27b.patch`           | git diff against `ds4.h`, `ds4.c`, `ds4_metal.m`    |
| `apply_qwen36_27b_patch.sh`  | wrapper that validates then applies the patch         |
| `qwen36_27b_config.h`        | architecture constants + `extern` config declaration |
| `qwen36_27b_metal.h`         | tuned Metal dispatch constants + function declaration |

## What the patch changes

### `ds4.h`

Adds the model configuration registry just before the closing `#endif`:

- `#define DS4_FILE_MAGIC_QWEN36_27B` on-disk file magic.
- `enum ds4_norm_type { DS4_NORM_LAYER, DS4_NORM_RMS }`.
- `enum ds4_activation { DS4_ACTIVATION_GELU, DS4_ACTIVATION_SWIGLU }`.
- `enum ds4_attention_type { DS4_ATTENTION_MHA, DS4_ATTENTION_GQA }`.
- `enum ds4_model_type { DS4_MODEL_QWEN36_27B, DS4_MODEL_COUNT }`.
- `struct ds4_model_config { ... }` and the lookup prototype
  `ds4_get_model_config()`.

### `ds4.c`

Appends the `qwen36_27b_config` definition (80 layers, 8192 hidden, 64 attn
heads, 8 KV heads / GQA, 22016 intermediate, 32768 max seq, 152064 vocab,
RoPE dim 128 / base 500000, RMSNorm, SwiGLU) and the `ds4_get_model_config()`
switch that returns it for `DS4_MODEL_QWEN36_27B`.

### `ds4_metal.m`

Appends `struct ds4_metal_config` and `configure_metal_for_model()`, which
populates sensible Metal dispatch defaults and, for the Qwen 3.6 27B file
magic, enables flash attention, SwiGLU, RMSNorm, RoPE and the model's rope dim.

## Model architecture

- Layers: 80
- Hidden size: 8192
- Attention heads: 64
- Key/Value heads: 8 (Grouped Query Attention)
- Intermediate size: 22016
- Max sequence length: 32768
- Vocabulary size: 152064
- RoPE dimension: 128
- RoPE base: 500000.0
- Normalization: RMSNorm
- MLP activation: SwiGLU
- Attention: Grouped Query Attention (GQA)

## Metal kernel support and execution routing

> **Status note (architectural phase).** The original patch only registered
> Qwen 3.6 27B *metadata* (the config registry + `configure_metal_for_model()`
> dispatch flags). That made the branch coherent and the patch applicable, but
> did not route tensors: `configure_metal_for_model()` populated flags only,
> and no execution path connected `DS4_MODEL_QWEN36_27B` to any Metal kernels.
> The DeepSeek/GLM kernels it claimed to reuse are MLA / indexer-specific and
> cannot express GQA (`n_kv_heads < n_head`). This commit adds the real route.

### What was actually missing (verified on the code)

- `kernel_glm_attention_full` (`metal/dsv4_misc.metal:1218`) indexes the KV
  cache as `s * n_head + head` -> assumes one KV row per query head (pure MHA).
  It cannot serve Qwen GQA (`n_head=64`, `n_kv_heads=8`).
- The `ds4_gpu_attention_*_heads_tensor` family is built around the DeepSeek
  MLA dual-cache (`raw_kv` + `comp_kv` + `comp_mask` + `sinks_offset`), not a
  plain GQA KV cache.
- The `glm_graph_forward_*` orchestrator in `ds4.c` is hard-wired to MLA
  low-rank Q (`attn_q_a`/`attn_q_b`), MLA KV (`attn_kv_a_mqa`/`attn_k_b`/
  `attn_v_b`) and the indexer; there was no standard-transformer (dense
  Q/K/V/O + SwiGLU FFN) forward.

### What this phase adds

| File                  | Change |
|-----------------------|--------|
| `metal/qwen_gqa.metal`| New file. `kernel_qwen_store_kv` stores K/V into a per-layer cache with `n_kv_heads` rows/position; `kernel_qwen_attention_gqa` runs causal grouped-query attention mapping each query head onto its owning KV head (`kv_head = head * n_kv_heads / n_head`). |
| `ds4_gpu.h`           | Declares `ds4_gpu_qwen_store_kv_tensor()` and `ds4_gpu_qwen_attention_gqa_tensor()`. |
| `ds4_metal.m`         | Implements the two tensor bindings (pipeline cache, command buffer, dispatch), registers + resets the pipelines, and rewrites `configure_metal_for_model()` so it *resolves* the `kernel_qwen_*` pipelines and exposes `qwen_gqa_pipelines_ready` (a real route flag, not just dispatch params). |
| `ds4.c`               | Adds `DS4_MODEL_FAMILY_QWEN36_27B`, `ds4_session_is_qwen()`, `weights_bind_qwen_layer()` (dense `attn_q/k/v/o` + `ffn_gate/up/down`), `ds4_qwen_gpu_graph` state, and `qwen_graph_forward_token()` -- a single-token decode graph that runs RMSNorm -> dense Q/K/V -> RoPE(Q,K) -> `qwen_store_kv` -> `qwen_attention_gqa` -> O projection -> residual -> RMSNorm -> SwiGLU(gate,up) -> down -> residual, per layer, then final norm + output projection. |

### Reused generic kernels (no changes needed)

- `ds4_gpu_matmul_q8_0_tensor`  - dense Q/K/V/O/gate/up/down projections
- `ds4_gpu_rms_norm_weight_tensor` - pre-attention / pre-FFN / final norm
- `ds4_gpu_rope_tail_tensor`   - RoPE on Q and K (in place, over `[n_head|n_kv_heads, head_dim]`)
- `ds4_gpu_swiglu_tensor`      - SwiGLU activation of the FFN gate/up pair
- `ds4_gpu_add_tensor`         - residual connections

### Session integration

`ds4_session` now owns the Qwen graph state end-to-end:

- `ds4_session_create()` allocates `s->qwen_graph` (scratch tensors + one
  `[cache_cap * n_kv_heads * head_dim]` KV cache per layer, `cache_cap = ctx_size`)
  and sets `s->qwen_graph_ready = true` when `DS4_MODEL_FAMILY_QWEN36_27B`.
- `ds4_session_eval()` runs `qwen_graph_forward_token()` (single-token decode)
  when `ds4_session_is_qwen(s)`, updating `pos = checkpoint.len`, the KV cache,
  and `s->logits`, before the GLM/DeepSeek/TP machinery is reached.
- `ds4_session_sync_internal()` (prefill) replays the prompt token-by-token
  through `ds4_session_eval()`; when the prompt extends a valid checkpoint only
  the new tail is replayed, otherwise KV state is reset and the prompt is
  replayed in full.
- `ds4_session_invalidate()` resets `qwen_graph.cache_len` on rewind.
- `ds4_session_free()` calls `qwen_graph_free()`.

### Limitations of this phase

- **Metal-only.** The CUDA backend is unchanged; GQA on CUDA is out of scope.
- **Single-token decode / prefill-by-replay.** There is no batched prefill
  kernel yet; prefill runs the decode path token-by-token. Correct, but not
  optimal for long prompts.
- **Not validated end-to-end here.** The sandbox is Linux without the Metal
  framework, so `ds4_metal.m` / `metal/qwen_gqa.metal` could not be compiled
  or run here. `ds4.c` / `ds4_gpu.h` were syntax-checked (see Verification).
  Build + smoke test against a real Qwen 3.6 27B GGUF must happen on macOS.

## Remaining work (real qwen35 model)

The kernels committed here target a *standard* dense GQA transformer and are
**insufficient for the real Qwen 3.6 27B** (hybrid linear/full attention).
Closing the gap is substantial new GPU work. Estimated components:

1. **Gated-DeltaNet linear-attention kernel (new).** ~48 of 64 layers use
   linear attention with a recurrent state matrix (no per-position KV cache).
   Needs: a per-token state-update kernel (delta rule + gate), a state-output
   matvec, and a per-layer recurrent-state tensor allocator in `ds4_session`
   (state shape ~ `[n_value_heads * value_head_dim, key_head_dim]`, reset on
   rewind). No such generic recurrent kernel exists in ds4 today; the closest
   precedent (`kernel_mul_mv_f16_f32_pair_compressor_store_4` in
   `metal/dense.metal`) is MLA-compressor-specific and not directly reusable.
   **Effort: largest single item.**
2. **Full-attention kernel extended to `head_dim 256` + partial rotary.** The
   16 full-attention layers use GQA but with `head_dim 256` (current kernel
   assumes 128) and `partial_rotary_factor 0.25` (rotate only 64 of 256 dims;
   current kernel rotates all). `kernel_qwen_attention_gqa` and the RoPE
   helper must be generalised.
3. **Per-layer-type dispatch.** `qwen_graph_forward_token` currently treats
   all layers identically. It must branch on `layer_types[i]`
   (`linear_attention` vs `full_attention`) and route to the right kernel,
   allocating KV cache only for the 16 full-attention layers.
4. **Multimodal handling.** The model is vision+text. For text-only inference
   the loader/graph must tolerate/ignore vision tensors and `image_token_id`
   rather than require them.
5. **Tensor-name binding.** `weights_bind_qwen_layer()` binds `attn_q/k/v/o`
   + `ffn_gate/up/down`. The qwen35 GGUF tensor names differ (linear-attention
   has `l_*` projections, conv kernel, gates) and must be re-bound against the
   real GGUF tensor list.
6. **GGUF metadata keys.** Confirm the exact `qwen35.*` key names from a real
   GGUF (this branch uses the standard `block_count` / `attention.*` /
   `rope.*` names, which match the upstream config but should be verified
   against the actual GGUF header).

A reasonable sequencing: confirm GGUF tensor names + metadata (item 6) first,
then item 3 + 2 (extend the existing full-attention path so the 16 GQA layers
work), then item 1 (the new linear-attention backend), then item 4/5 cleanup.


## macOS validation

The sandbox used for development is Linux x86_64 without the Metal framework,
so the Metal sources (`ds4_metal.m`, `metal/qwen_gqa.metal`) cannot be compiled
or run there. End-to-end validation must happen on a Mac with a Metal-capable
GPU and a real Qwen 3.6 27B GGUF.

The loader now recognises a Qwen3-architecture GGUF: `config_validate_model()`
branches on `general.architecture == "qwen3"`, calls
`config_validate_qwen_model()` which sets `g_ds4_shape = DS4_SHAPE_QWEN36_27B`,
and validates the GGUF keys (`qwen3.block_count`, `qwen3.embedding_length`,
`qwen3.attention.head_count`, `qwen3.attention.head_count_kv`,
`qwen3.attention.key_length`, `qwen3.rope.dimension_count`,
`qwen3.rope.freq_base`, `qwen3.attention.layer_norm_rms_epsilon`,
`qwen3.feed_forward_length`, `qwen3.vocab_size`, `qwen3.context_length`)
against the fixed shape table.

Steps on macOS:

```sh
# 1. Build the Metal binary.
make ds4

# 2. Confirm the GGUF is recognised (prints "arch:  qwen3" and the shape).
./ds4 -m models/Qwen3.6-27B-Q4_K_M.gguf -i </dev/null 2>&1 | head -20

# 3. Smoke gate: greedy generation, output must be non-empty + non-corrupt.
DS4_QWEN_MODEL=models/Qwen3.6-27B-Q4_K_M.gguf \
DS4_QWEN_CTX=4096 \
DS4_QWEN_GEN=48 \
./tests/qwen_metal_smoke.sh

# or via make:
make test-qwen-metal-smoke
```

The smoke gate (`tests/qwen_metal_smoke.sh`, modelled on
`tests/glm_long_context_smoke.sh`) runs greedy generation (`--temp 0`) and fails
on empty output, non-printable bytes, or known stream-corruption markers. It
exercises the full GQA path: `kernel_qwen_store_kv` + `kernel_qwen_attention_gqa`
KV cache, the per-layer KV allocation in `qwen_graph_alloc()`, and the
`qwen_graph_forward_token()` decode loop wired into `ds4_session_eval()`.

## Applying

```sh
./apply_qwen36_27b_patch.sh          # validate + apply
./apply_qwen36_27b_patch.sh --check  # validate only
```

or directly:

```sh
git apply qwen36_27b.patch
```

> Note: `qwen36_27b.patch` records the *metadata* phase only. The
> architectural phase (kernel + graph forward) lives directly in the source
> tree (`metal/qwen_gqa.metal`, `ds4_metal.m`, `ds4.c`, `ds4_gpu.h`) and is
> committed on this branch rather than carried as a patch file.

## Testing

`ds4_get_model_config(DS4_MODEL_QWEN36_27B)` returns the config and
`configure_metal_for_model()` resolves the GQA pipelines and sets
`qwen_gqa_pipelines_ready`. End-to-end decode requires macOS + a Qwen 3.6 27B
GGUF (see Limitations).
