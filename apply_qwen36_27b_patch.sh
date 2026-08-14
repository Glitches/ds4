#!/bin/bash
set -e

echo "=== Applying Qwen 3.6 27B patch ==="
echo ""

# Check if we're in the ds4 directory
if [ ! -f "ds4.h" ]; then
    echo "ERROR: Not in ds4 directory or ds4.h not found"
    exit 1
fi

echo "1. Backing up original files..."
cp ds4.h ds4.h.bak
cp ds4.c ds4.c.bak 2>/dev/null || echo "ds4.c not found, skipping"
cp ds4_metal.m ds4_metal.m.bak 2>/dev/null || echo "ds4_metal.m not found, skipping"
echo "   Backups created: ds4.h.bak, ds4.c.bak, ds4_metal.m.bak"
echo ""

echo "2. Modifying ds4.h..."

# Add DS4_MODEL_QWEN36_27B to the enum
sed -i.bak2 '/DS4_MODEL_COUNT/i    DS4_MODEL_QWEN36_27B,' ds4.h

# Add file magic definition
sed -i.bak2 '/DS4_FILE_MAGIC_[A-Z]/a#define DS4_FILE_MAGIC_QWEN36_27B 0xQW3627B0' ds4.h

# Add extern declaration at the end
echo "" >> ds4.h
echo "// Qwen 3.6 27B Model Support" >> ds4.h
echo "extern const struct ds4_model_config qwen36_27b_config;" >> ds4.h

echo "   Done ds4.h"
echo ""

echo "3. Modifying ds4.c..."

# Add case to switch statement
sed -i.bak2 '/case DS4_MODEL_[A-Z]*:/a    case DS4_MODEL_QWEN36_27B:
        return &qwen36_27b_config;' ds4.c

# Add model configuration at the end
cat >> ds4.c << 'EOF'

// Qwen 3.6 27B Model Configuration
static const struct ds4_model_config qwen36_27b_config = {
    .num_layers = 80,
    .hidden_size = 8192,
    .num_attention_heads = 64,
    .num_key_value_heads = 8,
    .intermediate_size = 22016,
    .max_seq_len = 32768,
    .vocab_size = 152064,
    .rope_dim = 128,
    .rope_base = 500000.0f,
    .norm_type = DS4_NORM_RMS,
    .mlp_activation = DS4_ACTIVATION_SWIGLU,
    .attention_type = DS4_ATTENTION_GQA,
    .model_name = "qwen36-27b",
    .file_id = DS4_FILE_MAGIC_QWEN36_27B,
    .quantization_version = 1,
};
EOF

echo "   Done ds4.c"
echo ""

echo "4. Modifying ds4_metal.m..."

# Add Metal configuration
sed -i.bak2 '/model_config->file_id == DS4_FILE_MAGIC_[A-Z]/a\n    // Qwen 3.6 27B Metal configuration
    if (model_config->file_id == DS4_FILE_MAGIC_QWEN36_27B) {
        metal_config.threads_per_threadgroup = 512;
        metal_config.max_threads_per_threadgroup = 1024;
        metal_config.threadgroup_size = MTLSizeMake(32, 16, 1);
        metal_config.use_flash_attention = true;
        metal_config.use_swiglu = true;
        metal_config.use_rmsnorm = true;
        metal_config.use_rope = true;
        metal_config.rope_dim = 128;
    }' ds4_metal.m

echo "   Done ds4_metal.m"
echo ""

echo "5. Cleaning up..."
rm -f ds4.h.bak2 ds4.c.bak2 ds4_metal.m.bak2

echo ""
echo "=== Patch applied! ==="
echo "Review with: git diff"
echo "Commit: git commit -am 'Add Qwen 3.6 27B Metal support'"
echo "Push: git push origin feat/qwen36-27b-metal"
