#pragma once

/*
 * Metal-specific configuration for Qwen/Qwen3.6-27B.
 *
 * The struct ds4_metal_config and configure_metal_for_model() are defined in
 * ds4_metal.m (added by qwen36_27b.patch). This header only exposes the
 * tuned Metal constants for Qwen 3.6 27B so they can be referenced without
 * pulling in the Objective-C translation unit.
 */

#ifndef QWEN36_27B_METAL_H
#define QWEN36_27B_METAL_H

#include "ds4.h"

/* Qwen 3.6 27B uses a standard transformer block (dense Q/K/V/O + SwiGLU FFN
 * + RMSNorm + RoPE, GQA with n_kv_heads=8). The MatMul / RMSNorm / SwiGLU /
 * RoPE stages reuse the existing generic Metal kernels, but grouped-query
 * attention is NOT covered by kernel_glm_attention_full (pure MHA) or the
 * DeepSeek MLA dual-cache attention family. The dedicated GQA kernels live in
 * metal/qwen_gqa.metal (kernel_qwen_store_kv, kernel_qwen_attention_gqa) and
 * are resolved by configure_metal_for_model(); see QWEN36_27B_PATCH.md. */

/* Tuned Metal dispatch parameters for Qwen 3.6 27B. */
#define QWEN36_27B_METAL_THREADS_PER_THREADGROUP      512
#define QWEN36_27B_METAL_MAX_THREADS_PER_THREADGROUP 1024
#define QWEN36_27B_METAL_THREADGROUP_W               32
#define QWEN36_27B_METAL_THREADGROUP_H               16
#define QWEN36_27B_METAL_THREADGROUP_D                1
#define QWEN36_27B_METAL_USE_FLASH_ATTENTION          true
#define QWEN36_27B_METAL_USE_SWIGLU                   true
#define QWEN36_27B_METAL_USE_RMSNORM                  true
#define QWEN36_27B_METAL_USE_ROPE                     true
#define QWEN36_27B_METAL_ROPE_DIM                     128

/* Declared in ds4_metal.m. Callers obtain a populated struct via
 * configure_metal_for_model() with the Qwen 3.6 27B model config. */
struct ds4_metal_config;
void configure_metal_for_model(struct ds4_metal_config *metal_config,
                               const struct ds4_model_config *model_config);

#endif /* QWEN36_27B_METAL_H */
