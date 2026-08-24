#pragma once

/*
 * Configuration for Qwen/Qwen3.6-27B model.
 * Architecture: https://huggingface.co/Qwen/Qwen3.6-27B
 *
 * The runtime struct (struct ds4_model_config) and the qwen36_27b_config
 * definition live in ds4.c (see the model configuration registry added by
 * qwen36_27b.patch). This header only exposes the architecture constants and
 * the extern declaration so callers can look the model up by type.
 */

#ifndef QWEN36_27B_CONFIG_H
#define QWEN36_27B_CONFIG_H

#include "ds4.h"

/* Qwen 3.6 27B architecture constants. */
#define QWEN36_27B_NUM_LAYERS          80
#define QWEN36_27B_HIDDEN_SIZE         8192
#define QWEN36_27B_NUM_ATTENTION_HEADS 64
#define QWEN36_27B_NUM_KEY_VALUE_HEADS 8
#define QWEN36_27B_INTERMEDIATE_SIZE   22016
#define QWEN36_27B_MAX_SEQ_LEN         32768
#define QWEN36_27B_VOCAB_SIZE          152064
#define QWEN36_27B_ROPE_DIM            128
#define QWEN36_27B_ROPE_BASE           500000.0f

/* Qwen 3.6 27B normalization / activation / attention selection. */
#define QWEN36_27B_NORM_TYPE           DS4_NORM_RMS
#define QWEN36_27B_MLP_ACTIVATION      DS4_ACTIVATION_SWIGLU
#define QWEN36_27B_ATTENTION_TYPE      DS4_ATTENTION_GQA

/* On-disk file magic. Defined in ds4.h. */
#define QWEN36_27B_FILE_ID             DS4_FILE_MAGIC_QWEN36_27B

/* Defined in ds4.c via the model configuration registry. */
extern const struct ds4_model_config qwen36_27b_config;

#endif /* QWEN36_27B_CONFIG_H */
