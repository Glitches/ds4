#pragma once

// Metal-specific configuration for Qwen/Qwen3.6-27B
// This file contains Metal kernel configurations specific to Qwen 3.6 27B

#ifndef QWEN36_27B_METAL_H
#define QWEN36_27B_METAL_H

#include "ds4_metal.h"

// Qwen 3.6 27B uses standard transformer blocks with the following Metal kernel requirements:
// - MatMul (dense.metal)
// - RMSNorm (norm.metal)
// - SwiGLU (glu.metal)
// - RoPE (rope.metal)
// - Flash Attention / Grouped Query Attention (flash_attn.metal)
// - Softmax (softmax.metal)

// Metal kernel configuration for Qwen 3.6 27B
// These values are optimized for the model's architecture
#define QWEN36_27B_METAL_THREADS_PER_THREADGROUP 512
#define QWEN36_27B_METAL_MAX_THREADS_PER_THREADGROUP 1024

// Qwen 3.6 27B specific Metal parameters
static const struct ds4_metal_config qwen36_27b_metal_config = {
    .threads_per_threadgroup = QWEN36_27B_METAL_THREADS_PER_THREADGROUP,
    .max_threads_per_threadgroup = QWEN36_27B_METAL_MAX_THREADS_PER_THREADGROUP,
    .threadgroup_size = {32, 16, 1},
    .use_flash_attention = true,
    .use_swiglu = true,
    .use_rmsnorm = true,
    .use_rope = true,
    .rope_dim = 128,
};

#endif // QWEN36_27B_METAL_H
