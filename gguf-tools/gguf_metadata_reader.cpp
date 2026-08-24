#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <unordered_map>
#include <cctype>

#pragma pack(push, 1)
struct GGUFHeader {
    char magic[4];      // "GGUF"
    uint32_t version;
    uint64_t n_tensors;
    uint64_t n_kv;
};
#pragma pack(pop)

// Helper functions for little-endian to host conversion
static uint32_t le_to_u32(const uint8_t* p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static uint64_t le_to_u64(const uint8_t* p) {
    return (uint64_t)p[0] | ((uint64_t)p[1] << 8) | ((uint64_t)p[2] << 16) | 
           ((uint64_t)p[3] << 24) | ((uint64_t)p[4] << 32) | ((uint64_t)p[5] << 40) | 
           ((uint64_t)p[6] << 48) | ((uint64_t)p[7] << 56);
}

static uint32_t read_u32_le(FILE* fp) {
    uint8_t b[4];
    if (fread(b, sizeof(b), 1, fp) != 1) {
        fprintf(stderr, "error: failed to read uint32\n");
        exit(1);
    }
    return le_to_u32(b);
}

static uint64_t read_u64_le(FILE* fp) {
    uint8_t b[8];
    if (fread(b, sizeof(b), 1, fp) != 1) {
        fprintf(stderr, "error: failed to read uint64\n");
        exit(1);
    }
    return le_to_u64(b);
}

static void read_bytes(FILE* fp, void* buf, size_t len, const char* what) {
    if (fread(buf, len, 1, fp) != 1) {
        fprintf(stderr, "error: failed to read %s (%zu bytes)\n", what, len);
        exit(1);
    }
}

enum class GGUFValueType : uint32_t {
    UINT8   = 0,
    INT8    = 1,
    UINT16  = 2,
    INT16   = 3,
    UINT32  = 4,
    INT32   = 5,
    FLOAT32 = 6,
    BOOL    = 7,
    STRING  = 8,
    ARRAY   = 9,
    UINT64  = 10,
    INT64   = 11,
    FLOAT64 = 12,
};

static std::string read_string(FILE* fp);
static void print_value(FILE* fp, GGUFValueType type, const std::string& key = "");

static std::string read_string(FILE* fp) {
    uint64_t len = read_u64_le(fp);
    
    // Sanity check: limit string length to 1MB
    if (len > 1024 * 1024) {
        fprintf(stderr, "error: string length too large (%lu bytes)\n", len);
        exit(1);
    }
    
    std::string s(len, '\0');
    if (len > 0) {
        read_bytes(fp, &s[0], len, "string data");
    }
    return s;
}

// Check if a byte array looks like printable ASCII/UTF-8 text
static bool is_printable_text(const uint8_t* data, size_t len) {
    if (len == 0) return false;
    bool has_printable = false;
    for (size_t i = 0; i < len; i++) {
        if (data[i] >= 32 && data[i] <= 126) {
            has_printable = true;
        } else if (data[i] != 0 && data[i] != '\t' && data[i] != '\n' && data[i] != '\r') {
            // Non-printable, non-null character
            return false;
        }
    }
    return has_printable;
}

// For UINT8 arrays that contain null-separated strings (like GGUF tags field)
static void print_uint8_array_as_strings(FILE* fp, uint64_t n, const std::string& key) {
    std::vector<uint8_t> bytes(n);
    if (fread(bytes.data(), n, 1, fp) != 1) {
        fprintf(stderr, "error: failed to read UINT8 array data\n");
        exit(1);
    }
    
    // Check if this looks like null-separated strings
    bool has_nulls = false;
    for (uint64_t i = 0; i < n; i++) {
        if (bytes[i] == 0) {
            has_nulls = true;
            break;
        }
    }
    
    if (has_nulls) {
        // Parse as null-separated strings
        printf("[");
        bool first = true;
        size_t start = 0;
        for (size_t i = 0; i <= n; i++) {
            if (i == n || bytes[i] == 0) {
                if (i > start) {
                    if (!first) printf(",");
                    printf("\"");
                    for (size_t j = start; j < i; j++) {
                        if (bytes[j] == '\"' || bytes[j] == '\\') {
                            printf("\\%c", bytes[j]);
                        } else if (bytes[j] >= 32 && bytes[j] <= 126) {
                            printf("%c", bytes[j]);
                        } else {
                            printf("\\x%02x", bytes[j]);
                        }
                    }
                    printf("\"");
                    first = false;
                }
                start = i + 1;
            }
        }
        printf("]");
    } else if (is_printable_text(bytes.data(), n)) {
        // Print as a single string
        printf("\"");
        for (uint64_t i = 0; i < n; i++) {
            if (bytes[i] == '\"' || bytes[i] == '\\') {
                printf("\\%c", bytes[i]);
            } else if (bytes[i] >= 32 && bytes[i] <= 126) {
                printf("%c", bytes[i]);
            } else {
                printf("\\x%02x", bytes[i]);
            }
        }
        printf("\"");
    } else {
        // Print as numbers
        printf("[");
        for (uint64_t i = 0; i < n; i++) {
            if (i > 0) printf(",");
            printf("%u", bytes[i]);
        }
        printf("]");
    }
}

static void print_array_of_strings(FILE* fp, uint64_t n) {
    printf("[");
    for (uint64_t i = 0; i < n; i++) {
        if (i > 0) printf(",");
        std::string s = read_string(fp);
        printf("\"%s\"", s.c_str());
    }
    printf("]");
}

static void print_array(FILE* fp, GGUFValueType elem_type, uint64_t n, const std::string& key = "") {
    // Special case: UINT8 array - might be null-separated strings
    if (elem_type == GGUFValueType::UINT8 && n > 0) {
        print_uint8_array_as_strings(fp, n, key);
        return;
    }
    
    // Generic array printing for other types
    printf("[");
    for (uint64_t i = 0; i < n; i++) {
        if (i > 0) printf(",");
        print_value(fp, elem_type, key);
    }
    printf("]");
}

static void print_float32(uint32_t bits) {
    union {
        uint32_t u;
        float f;
    } v;
    v.u = bits;
    printf("%f", v.f);
}

static void print_float64(uint64_t bits) {
    union {
        uint64_t u;
        double f;
    } v;
    v.u = bits;
    printf("%lf", v.f);
}

static void print_value(FILE* fp, GGUFValueType type, const std::string& key) {
    switch (type) {
        case GGUFValueType::UINT8: {
            uint8_t v;
            read_bytes(fp, &v, sizeof(v), "UINT8 value");
            printf("%u", v);
            break;
        }
        case GGUFValueType::INT8: {
            int8_t v;
            read_bytes(fp, &v, sizeof(v), "INT8 value");
            printf("%d", v);
            break;
        }
        case GGUFValueType::UINT16: {
            uint16_t v;
            uint8_t b[2];
            read_bytes(fp, b, sizeof(b), "UINT16 value");
            v = (uint16_t)b[0] | ((uint16_t)b[1] << 8);
            printf("%u", v);
            break;
        }
        case GGUFValueType::INT16: {
            int16_t v;
            uint8_t b[2];
            read_bytes(fp, b, sizeof(b), "INT16 value");
            v = (int16_t)(b[0] | (b[1] << 8));
            printf("%d", v);
            break;
        }
        case GGUFValueType::UINT32: {
            uint32_t v = read_u32_le(fp);
            printf("%u", v);
            break;
        }
        case GGUFValueType::INT32: {
            uint32_t bits = read_u32_le(fp);
            printf("%d", (int32_t)bits);
            break;
        }
        case GGUFValueType::FLOAT32: {
            uint32_t bits = read_u32_le(fp);
            print_float32(bits);
            break;
        }
        case GGUFValueType::BOOL: {
            uint8_t v;
            read_bytes(fp, &v, sizeof(v), "BOOL value");
            printf("%s", v ? "true" : "false");
            break;
        }
        case GGUFValueType::STRING: {
            std::string s = read_string(fp);
            printf("%s", s.c_str());
            break;
        }
        case GGUFValueType::ARRAY: {
            uint64_t n = read_u64_le(fp);
            uint32_t elem_type_raw = read_u32_le(fp);
            GGUFValueType elem_type = static_cast<GGUFValueType>(elem_type_raw);
            
            // Special case: array of strings stored as [uint64 len][string data]...
            if (elem_type == GGUFValueType::STRING) {
                print_array_of_strings(fp, n);
            } else {
                print_array(fp, elem_type, n, key);
            }
            break;
        }
        case GGUFValueType::UINT64: {
            uint64_t v = read_u64_le(fp);
            printf("%lu", v);
            break;
        }
        case GGUFValueType::INT64: {
            uint64_t bits = read_u64_le(fp);
            printf("%ld", (int64_t)bits);
            break;
        }
        case GGUFValueType::FLOAT64: {
            uint64_t bits = read_u64_le(fp);
            print_float64(bits);
            break;
        }
    }
}

static void read_metadata(FILE* fp, uint64_t n_kv) {
    for (uint64_t i = 0; i < n_kv; i++) {
        std::string key = read_string(fp);
        uint32_t type_raw = read_u32_le(fp);
        GGUFValueType type = static_cast<GGUFValueType>(type_raw);
        printf("%s=", key.c_str());
        print_value(fp, type, key);
        printf("\n");
    }
}

int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <gguf-file>\n", argv[0]);
        return 1;
    }

    FILE* fp = fopen(argv[1], "rb");
    if (!fp) {
        fprintf(stderr, "error: cannot open file '%s'\n", argv[1]);
        return 1;
    }

    GGUFHeader header;
    if (fread(&header, sizeof(header), 1, fp) != 1) {
        fprintf(stderr, "error: failed to read GGUF header\n");
        fclose(fp);
        return 1;
    }

    if (memcmp(header.magic, "GGUF", 4) != 0) {
        fprintf(stderr, "error: not a GGUF file (magic: %c%c%c%c)\n",
                header.magic[0], header.magic[1], header.magic[2], header.magic[3]);
        fclose(fp);
        return 1;
    }

    // Convert header fields from little-endian
    uint32_t version = le_to_u32(reinterpret_cast<uint8_t*>(&header.version));
    uint64_t n_tensors = le_to_u64(reinterpret_cast<uint8_t*>(&header.n_tensors));
    uint64_t n_kv = le_to_u64(reinterpret_cast<uint8_t*>(&header.n_kv));

    // Print header info
    printf("gguf.magic=GGUF\n");
    printf("gguf.version=%u\n", version);
    printf("gguf.tensor_count=%lu\n", n_tensors);
    printf("gguf.kv_count=%lu\n", n_kv);

    // Read metadata
    read_metadata(fp, n_kv);

    fclose(fp);
    return 0;
}
