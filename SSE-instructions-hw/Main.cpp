#include <immintrin.h>

struct vector4 {
private:
    __m128 data;

public:
    vector4(float x, float y, float z) : data(_mm_set_ps(x, y, z, 0.0f)) {}
    vector4(float x, float y, float z, float w) : data(_mm_set_ps(x, y, z, w)) {}

    vector4& add(const vector4& other) {
        data = _mm_add_ps(data, other.data);
        return *this;
    }
    vector4& add(float x, float y, float z) {
        __m128 other = _mm_set_ps(x, y, z, 0.0f);
        data = _mm_add_ps(data, other);
        return *this;
    }

    vector4& sub(const vector4& other) {
        data = _mm_sub_ps(data, other.data);
        return *this;
    }
    vector4& sub(float x, float y, float z) {
        __m128 other = _mm_set_ps(x, y, z, 0.0f);
        data = _mm_sub_ps(data, other);
        return *this;
    }

    vector4& mul(float scale) {
        __m128 scalar = _mm_set1_ps(scale);
        data = _mm_mul_ps(data, scalar);
        return *this;
    }
    vector4& mul(float scale, float w_scale) {
        __m128 scalar = _mm_set_ps(scale, scale, scale, w_scale);
        data = _mm_mul_ps(data, scalar);
        return *this;
    }

    vector4& div(float scale) {
        __m128 scalar = _mm_set1_ps(scale);
        data = _mm_div_ps(data, scalar);
        return *this;
    }
    vector4& div(float scale, float w_scale) {
        __m128 scalar = _mm_set_ps(scale, scale, scale, w_scale);
        data = _mm_div_ps(data, scalar);
        return *this;
    }
};
