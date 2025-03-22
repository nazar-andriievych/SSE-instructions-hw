#include <immintrin.h>

struct vector4 {
private:
    __m128 data;

public:
    vector4(float x, float y, float z) : data(_mm_set_ps(x, y, z, 0.0f)) {}
    vector4(float x, float y, float z, float w) : data(_mm_set_ps(x, y, z, w)) {}
};
