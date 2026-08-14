#pragma once

// Configuration for Qwen/Qwen3.6-27B model
// Based on the architecture described at: https://huggingface.co/Qwen/Qwen3.6-27B

#ifndef QWEN36_27B_CONFIG_H
#define QWEN36_27B_CONFIG_H

#include "ds4.h"

// Qwen 3.6 27B Model Architecture
#define QWEN36_27B_NUM_LAYERS 80
#define QWEN36_27B_HIDDEN_SIZE 8192
#define QWEN36_27B_NUM_ATTENTION_HEADS 64
#define QWEN36_27B_NUM_KEY_VALUE_HEADS 8
#define QWEN36_27B_INTERMEDIATE_SIZE 22016
#define QWEN36_27B_MAX_SEQ_LEN 32768
#define QWEN36_27B_VOCAB_SIZE 152064
#define QWEN36_27B_ROPE_DIM 128
#define QWEN36_27B_ROPE_BASE 500000.0f

// Qwen 3.6 uses RMSNorm
#define QWEN36_27B_NORM_TYPE DS4_NORM_RMS

// Qwen 3.6 uses SwiGLU activation for MLP
#define QWEN36_27B_MLP_ACTIVATION DS4_ACTIVATION_SWIGLU

// Qwen 3.6 uses Grouped Query Attention (GQA)
#define QWEN36_27B_ATTENTION_TYPE DS4_ATTENTION_GQA

// Model configuration structure
static const struct ds4_model_config qwen36_27b_config = {
    .num_layers = QWEN36_27B_NUM_LAYERS,
    .hidden_size = QWEN36_27B_HIDDEN_SIZE,
    .num_attention_heads = QWEN36_27B_NUM_ATTENTION_HEADS,
    .num_key_value_heads = QWEN36_27B_NUM_KEY_VALUE_HEADS,
    .intermediate_size = QWEN36_27B_INTERMEDIATE_SIZE,
    .max_seq_len = QWEN36_27B_MAX_SEQ_LEN,
    .vocab_size = QWEN36_27B_VOCAB_SIZE,
    .rope_dim = QWEN36_27B_ROPE_DIM,
    .rope_base = QWEN36_27B_ROPE_BASE,
    .norm_type = QWEN36_27B_NORM_TYPE,
    .mlp_activation = QWEN36_27B_MLP_ACTIVATION,
    .attention_type = QWEN36_27B_ATTENTION_TYPE,
    .model_name = "qwen36-27b",
    .file_id = DS4_FILE_MAGIC_QWEN36_27B,
    .quantization_version = 1,
};

#endif // QWEN36_27B_CONFIG_H
