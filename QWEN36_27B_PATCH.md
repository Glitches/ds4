# Patch for Qwen 3.6 27B Support in ds4

## Overview
This patch adds support for Qwen/Qwen3.6-27B model with Metal backend to ds4.

## Files Added
1. `qwen36_27b_config.h` - Model configuration header
2. `qwen36_27b_metal.h` - Metal-specific configuration header

## Files to Modify

### 1. ds4.h

#### Add Model Type Enum
Find the `ds4_model_type` or `enum ds4_model` definition and add:
```c
DS4_MODEL_QWEN36_27B,
```

Before the closing of the enum (usually before `DS4_MODEL_COUNT` or similar).

#### Add File Magic
Find the file magic definitions and add:
```c
DS4_FILE_MAGIC_QWEN36_27B = 0xQW3627B0,
```

#### Add Model Configuration
Find where other model configurations are declared (look for `static const struct ds4_model_config` for other models) and add:
```c
extern const struct ds4_model_config qwen36_27b_config;
```

### 2. ds4.c or ds4_model.c

#### Register Model Configuration
Find the model registration section and add:
```c
case DS4_MODEL_QWEN36_27B:
    return &qwen36_27b_config;
```

Or if using a registration macro:
```c
DS4_MODEL_REGISTER(QWEN36_27B, qwen36_27b_config);
```

### 3. ds4_metal.m

#### Add Metal Configuration
Find the Metal configuration setup (look for `ds4_metal_config` or similar) and add:
```objectivec
if (model_config->file_id == DS4_FILE_MAGIC_QWEN36_27B) {
    metal_config.threads_per_threadgroup = 512;
    metal_config.max_threads_per_threadgroup = 1024;
    metal_config.threadgroup_size = MTLSizeMake(32, 16, 1);
    metal_config.use_flash_attention = true;
    metal_config.use_swiglu = true;
    metal_config.use_rmsnorm = true;
    metal_config.use_rope = true;
    metal_config.rope_dim = 128;
}
```

## Model Architecture Details

Qwen 3.6 27B uses the following architecture:
- **Layers**: 80
- **Hidden Size**: 8192
- **Attention Heads**: 64
- **Key/Value Heads**: 8 (Grouped Query Attention)
- **Intermediate Size**: 22016
- **Max Sequence Length**: 32768
- **Vocabulary Size**: 152064
- **RoPE Dimension**: 128
- **RoPE Base**: 500000.0
- **Normalization**: RMSNorm
- **MLP Activation**: SwiGLU
- **Attention**: Grouped Query Attention (GQA)

## Metal Kernel Support

The existing Metal kernels in the `metal/` directory already support all required operations:
- `dense.metal` - Matrix multiplication
- `norm.metal` - RMSNorm
- `glu.metal` - SwiGLU activation
- `rope.metal` - Rotary Position Embedding
- `flash_attn.metal` - Flash Attention / Grouped Query Attention
- `softmax.metal` - Softmax

No new Metal kernels need to be created for this model.

## Testing

After applying this patch:
1. The model should be selectable with `--model qwen36-27b` or similar
2. Metal backend should work on macOS with Apple Silicon
3. The model should load and run with the specified architecture parameters

## References
- Model: https://huggingface.co/Qwen/Qwen3.6-27B
- Original ds4: https://github.com/antirez/ds4