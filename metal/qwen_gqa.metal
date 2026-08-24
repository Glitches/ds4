/*
 * Qwen standard-transformer GQA kernels.
 *
 * Unlike kernel_glm_attention_full (which assumes one KV row per Q head, i.e.
 * pure MHA), these kernels keep a KV cache with n_kv_heads rows per sequence
 * position and map each of the n_head query heads onto its owning KV head via
 * kv_head = head / (n_head / n_kv_heads). This is the routing the existing
 * Metal kernels built for DeepSeek MLA / GLM indexer cannot express.
 *
 * Used by the Qwen 3.6 27B graph forward in ds4.c. The matmul/RMSNorm/RoPE/
 * SwiGLU stages reuse the existing generic tensor kernels; only KV store and
 * grouped-query attention need dedicated implementations here.
 */

#ifndef DS4_METAL_QWEN_GQA_H
#define DS4_METAL_QWEN_GQA_H

struct ds4_metal_args_qwen_store_kv {
    uint32_t pos0;        /* first sequence position written by this launch */
    uint32_t n_tokens;    /* number of token rows in the source k/v tensors */
    uint32_t cache_cap;   /* row capacity of the key/value caches */
    uint32_t n_kv_heads;  /* number of KV heads stored in the caches */
    uint32_t head_dim;    /* per-head dimensionality of K and V */
    uint32_t cache_f16;   /* 1 -> half, 0 -> float cache element */
    uint32_t pad0;
    uint32_t pad1;
};

struct ds4_metal_args_qwen_attention {
    uint32_t pos0;        /* first query position */
    uint32_t n_tokens;    /* number of query rows */
    uint32_t cache_len;   /* number of valid cache rows (>= pos0 + n_tokens) */
    uint32_t cache_cap;   /* row capacity of the key/value caches */
    uint32_t n_head;      /* number of query heads */
    uint32_t n_kv_heads;  /* number of KV heads stored in the caches */
    uint32_t head_dim;    /* per-head dimensionality of Q, K, V */
    uint32_t cache_f16;   /* 1 -> half, 0 -> float cache element */
    uint32_t pad0;
    uint32_t pad1;
    float    scale;       /* typically 1/sqrt(head_dim) */
};

/* Forward-declared helpers mirrored from dsv4_misc.metal so this translation
 * unit stays self-contained. Each .metal file is compiled independently. */
static inline float qwen_cache_load_f32_or_f16(
        device const char *base,
        uint64_t index,
        uint cache_f16) {
    if (cache_f16 != 0u) {
        return (float)((device const half *)base)[index];
    }
    return ((device const float *)base)[index];
}

static inline float4 qwen_cache_load4_f32_or_f16(
        device const char *base,
        uint64_t index,
        uint cache_f16) {
    if (cache_f16 != 0u) {
        device const half *h = (device const half *)base;
        return float4((float)h[index + 0u],
                      (float)h[index + 1u],
                      (float)h[index + 2u],
                      (float)h[index + 3u]);
    }
    return ((device const float4 *)base)[index >> 2u];
}

static inline void qwen_cache_store_f32_or_f16(
        device char *base,
        uint64_t index,
        float value,
        uint cache_f16) {
    if (cache_f16 != 0u) {
        ((device half *)base)[index] = (half)value;
    } else {
        ((device float *)base)[index] = value;
    }
}

/*
 * Store n_tokens rows of K and V (each row: n_kv_heads * head_dim) into the
 * per-layer KV caches at positions [pos0, pos0 + n_tokens). RoPE is assumed to
 * have already been applied to K on the host side (via ds4_gpu_rope_tail_tensor
 * operating on the packed [n_kv_heads, head_dim] layout). One threadgroup
 * row writes one (token, kv_head, head_dim) tile of K and V.
 */
kernel void kernel_qwen_store_kv(
        constant ds4_metal_args_qwen_store_kv & args,
        device const char *k_src,
        device const char *v_src,
        device       char *key_cache,
        device       char *value_cache,
        uint3 tgpig [[threadgroup_position_in_grid]],
        uint  tid   [[thread_index_in_threadgroup]],
        uint3 ntg   [[threads_per_threadgroup]]) {
    const uint token = tgpig.x;
    const uint head  = tgpig.y;
    if (token >= args.n_tokens || head >= args.n_kv_heads) return;
    const uint nth   = ntg.x;
    const uint row   = args.pos0 + token;
    if (row >= args.cache_cap) return;

    const uint64_t src_off =
        ((uint64_t)token * args.n_kv_heads + head) * args.head_dim;
    const uint64_t dst_off =
        ((uint64_t)row   * args.n_kv_heads + head) * args.head_dim;

    for (uint d = tid; d < args.head_dim; d += nth) {
        qwen_cache_store_f32_or_f16(
            key_cache, dst_off + d,
            qwen_cache_load_f32_or_f16(k_src, src_off + d, args.cache_f16),
            args.cache_f16);
        qwen_cache_store_f32_or_f16(
            value_cache, dst_off + d,
            qwen_cache_load_f32_or_f16(v_src, src_off + d, args.cache_f16),
            args.cache_f16);
    }
}

/*
 * Grouped-query attention, causal, for a standard transformer block.
 *
 * Grid: (n_tokens, n_head). Each threadgroup computes one (token, head) output
 * row. The KV cache holds n_kv_heads rows per position; query head `head` reads
 * KV head `kv_head = head * n_kv_heads / n_head`. Scores are computed over the
 * visible causal window [0, pos0 + token] and softmax-weighted into the output.
 *
 * scratch must hold (256 + cache_cap) floats; the caller sizes threadgroup
 * memory accordingly (see ds4_gpu_qwen_attention_gqa_tensor).
 */
kernel void kernel_qwen_attention_gqa(
        constant ds4_metal_args_qwen_attention & args,
        device const char *q,
        device const char *key_cache,
        device const char *value_cache,
        device       char *heads,
        threadgroup float *scratch [[threadgroup(0)]],
        uint tid   [[thread_index_in_threadgroup]],
        uint3 ntg_u [[threads_per_threadgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    const uint token = tgpig.x;
    const uint head  = tgpig.y;
    if (token >= args.n_tokens || head >= args.n_head) return;
    const uint nth = ntg_u.x;
    const uint hd4 = args.head_dim / 4u;
    const uint visible = min(args.cache_len, args.pos0 + token + 1u);
    const uint kv_head = head * args.n_kv_heads / args.n_head;

    threadgroup float *red    = scratch;
    threadgroup float *scores = scratch + 256u;

    /* Q vector for this (token, head). */
    device const float4 *q4 = (device const float4 *)(q +
        ((uint64_t)token * args.n_head + head) * args.head_dim * sizeof(float));

    /* Score each visible KV row against Q (dot product, scaled). */
    for (uint s = tid; s < visible; s += nth) {
        const uint64_t kbase =
            ((uint64_t)s * args.n_kv_heads + kv_head) * args.head_dim;
        float dotv = 0.0f;
        for (uint i = 0; i < hd4; i++) {
            dotv += dot(q4[i],
                        qwen_cache_load4_f32_or_f16(key_cache,
                                                    kbase + 4u * (uint64_t)i,
                                                    args.cache_f16));
        }
        scores[s] = dotv * args.scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    /* Softmax (single-thread reduction; visible <= cache_cap is bounded). */
    if (tid == 0u) {
        float max_score = -INFINITY;
        for (uint s = 0; s < visible; s++) {
            max_score = max(max_score, scores[s]);
        }
        float sum = 0.0f;
        for (uint s = 0; s < visible; s++) {
            const float w = exp(scores[s] - max_score);
            scores[s] = w;
            sum += w;
        }
        red[0] = max(sum, 1.0e-20f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    /* Weighted sum of V rows -> output. */
    const float denom = red[0];
    device float *out = (device float *)(heads +
        ((uint64_t)token * args.n_head + head) * args.head_dim * sizeof(float));
    for (uint d = tid; d < args.head_dim; d += nth) {
        float acc = 0.0f;
        for (uint s = 0; s < visible; s++) {
            const uint64_t vbase =
                ((uint64_t)s * args.n_kv_heads + kv_head) * args.head_dim;
            acc += scores[s] *
                   qwen_cache_load_f32_or_f16(value_cache,
                                             vbase + d,
                                             args.cache_f16);
        }
        out[d] = acc / denom;
    }
}

#endif /* DS4_METAL_QWEN_GQA_H */
