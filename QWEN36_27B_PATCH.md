# Qwen 3.6 27B support in ds4

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

## Metal kernel support

The existing Metal kernels in `metal/` already cover every operation Qwen
3.6 27B needs; no new kernels are created:

- `dense.metal` - matrix multiplication
- `norm.metal` - RMSNorm
- `glu.metal` - SwiGLU activation
- `rope.metal` - rotary position embedding
- `flash_attn.metal` - flash attention / grouped query attention
- `softmax.metal` - softmax

## Applying

```sh
./apply_qwen36_27b_patch.sh          # validate + apply
./apply_qwen36_27b_patch.sh --check  # validate only
```

or directly:

```sh
git apply qwen36_27b.patch
```

## Testing

After applying the patch the model config is reachable through
`ds4_get_model_config(DS4_MODEL_QWEN36_27B)`, and Metal dispatch parameters
through `configure_metal_for_model()`.
