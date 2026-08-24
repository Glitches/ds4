#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <unordered_map>

#pragma pack(push, 1)
struct GGUFHeader {
    char magic[4];      // "GGUF"
    uint32_t version;
    uint64_t n_tensors;
    uint64_t n_kv;
};
#pragma pack(pop)

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

static std::string type_to_string(GGUFValueType type) {
    switch (type) {
        case GGUFValueType::UINT8:   return "UINT8";
        case GGUFValueType::INT8:    return "INT8";
        case GGUFValueType::UINT16:  return "UINT16";
        case GGUFValueType::INT16:   return "INT16";
        case GGUFValueType::UINT32:  return "UINT32";
        case GGUFValueType::INT32:   return "INT32";
        case GGUFValueType::FLOAT32: return "FLOAT32";
        case GGUFValueType::BOOL:    return "BOOL";
        case GGUFValueType::STRING:  return "STRING";
        case GGUFValueType::ARRAY:   return "ARRAY";
        case GGUFValueType::UINT64:  return "UINT64";
        case GGUFValueType::INT64:   return "INT64";
        case GGUFValueType::FLOAT64: return "FLOAT64";
        default: return "UNKNOWN";
    }
}

static std::string read_string(FILE* fp) {
    uint64_t len;
    if (fread(&len, sizeof(len), 1, fp) != 1) {
        fprintf(stderr, "error: failed to read string length\n");
        exit(1);
    }
    std::string s(len, '\0');
    if (len > 0 && fread(&s[0], len, 1, fp) != 1) {
        fprintf(stderr, "error: failed to read string data\n");
        exit(1);
    }
    return s;
}

static void skip_value(FILE* fp, GGUFValueType type) {
    switch (type) {
        case GGUFValueType::UINT8: {
            uint8_t v;
            fread(&v, sizeof(v), 1, fp);
            break;
        }
        case GGUFValueType::INT8: {
            int8_t v;
            fread(&v, sizeof(v), 1, fp);
            break;
        }
        case GGUFValueType::UINT16: {
            uint16_t v;
            fread(&v, sizeof(v), 1, fp);
            break;
        }
        case GGUFValueType::INT16: {
            int16_t v;
            fread(&v, sizeof(v), 1, fp);
            break;
        }
        case GGUFValueType::UINT32: {
            uint32_t v;
            fread(&v, sizeof(v), 1, fp);
            break;
        }
        case GGUFValueType::INT32: {
            int32_t v;
            fread(&v, sizeof(v), 1, fp);
            break;
        }
        case GGUFValueType::FLOAT32: {
            float v;
            fread(&v, sizeof(v), 1, fp);
            break;
        }
        case GGUFValueType::BOOL: {
            uint8_t v;
            fread(&v, sizeof(v), 1, fp);
            break;
        }
        case GGUFValueType::STRING: {
            read_string(fp);
            break;
        }
        case GGUFValueType::ARRAY: {
            uint64_t n;
            fread(&n, sizeof(n), 1, fp);
            GGUFValueType elem_type;
            fread(&elem_type, sizeof(elem_type), 1, fp);
            for (uint64_t i = 0; i < n; i++) {
                skip_value(fp, elem_type);
            }
            break;
        }
        case GGUFValueType::UINT64: {
            uint64_t v;
            fread(&v, sizeof(v), 1, fp);
            break;
        }
        case GGUFValueType::INT64: {
            int64_t v;
            fread(&v, sizeof(v), 1, fp);
            break;
        }
        case GGUFValueType::FLOAT64: {
            double v;
            fread(&v, sizeof(v), 1, fp);
            break;
        }
    }
}

static void print_value(FILE* fp, GGUFValueType type) {
    switch (type) {
        case GGUFValueType::UINT8: {
            uint8_t v;
            fread(&v, sizeof(v), 1, fp);
            printf("%u", v);
            break;
        }
        case GGUFValueType::INT8: {
            int8_t v;
            fread(&v, sizeof(v), 1, fp);
            printf("%d", v);
            break;
        }
        case GGUFValueType::UINT16: {
            uint16_t v;
            fread(&v, sizeof(v), 1, fp);
            printf("%u", v);
            break;
        }
        case GGUFValueType::INT16: {
            int16_t v;
            fread(&v, sizeof(v), 1, fp);
            printf("%d", v);
            break;
        }
        case GGUFValueType::UINT32: {
            uint32_t v;
            fread(&v, sizeof(v), 1, fp);
            printf("%u", v);
            break;
        }
        case GGUFValueType::INT32: {
            int32_t v;
            fread(&v, sizeof(v), 1, fp);
            printf("%d", v);
            break;
        }
        case GGUFValueType::FLOAT32: {
            float v;
            fread(&v, sizeof(v), 1, fp);
            printf("%f", v);
            break;
        }
        case GGUFValueType::BOOL: {
            uint8_t v;
            fread(&v, sizeof(v), 1, fp);
            printf("%s", v ? "true" : "false");
            break;
        }
        case GGUFValueType::STRING: {
            std::string s = read_string(fp);
            printf("%s", s.c_str());
            break;
        }
        case GGUFValueType::ARRAY: {
            uint64_t n;
            fread(&n, sizeof(n), 1, fp);
            GGUFValueType elem_type;
            fread(&elem_type, sizeof(elem_type), 1, fp);
            printf("[");
            for (uint64_t i = 0; i < n; i++) {
                if (i > 0) printf(",");
                print_value(fp, elem_type);
            }
            printf("]");
            break;
        }
        case GGUFValueType::UINT64: {
            uint64_t v;
            fread(&v, sizeof(v), 1, fp);
            printf("%lu", v);
            break;
        }
        case GGUFValueType::INT64: {
            int64_t v;
            fread(&v, sizeof(v), 1, fp);
            printf("%ld", v);
            break;
        }
        case GGUFValueType::FLOAT64: {
            double v;
            fread(&v, sizeof(v), 1, fp);
            printf("%lf", v);
            break;
        }
    }
}

static void read_metadata(FILE* fp, uint64_t n_kv) {
    for (uint64_t i = 0; i < n_kv; i++) {
        std::string key = read_string(fp);
        uint32_t type_raw;
        if (fread(&type_raw, sizeof(type_raw), 1, fp) != 1) {
            fprintf(stderr, "error: failed to read KV type\n");
            exit(1);
        }
        GGUFValueType type = static_cast<GGUFValueType>(type_raw);
        printf("general.%s=", key.c_str());
        print_value(fp, type);
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

    // Print header info
    printf("gguf.magic=GGUF\n");
    printf("gguf.version=%u\n", header.version);
    printf("gguf.tensor_count=%lu\n", header.n_tensors);
    printf("gguf.kv_count=%lu\n", header.n_kv);

    // Read metadata
    read_metadata(fp, header.n_kv);

    fclose(fp);
    return 0;
}
