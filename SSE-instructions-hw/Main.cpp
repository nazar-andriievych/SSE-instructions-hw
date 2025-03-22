#include <immintrin.h>
#include <cmath>

struct vector4 {
private:
    __m128 data;

public:
    vector4(float x, float y, float z)
        : data(_mm_setr_ps(x, y, z, 0.0f)) {
    }
    vector4(float x, float y, float z, float w)
        : data(_mm_setr_ps(x, y, z, w)) {
    }

    float x() const { return _mm_cvtss_f32(data); }
    float y() const { return _mm_cvtss_f32(_mm_shuffle_ps(data, data, _MM_SHUFFLE(1, 1, 1, 1))); }
    float z() const { return _mm_cvtss_f32(_mm_shuffle_ps(data, data, _MM_SHUFFLE(2, 2, 2, 2))); }
    float w() const { return _mm_cvtss_f32(_mm_shuffle_ps(data, data, _MM_SHUFFLE(3, 3, 3, 3))); }

    vector4& add(const vector4& other) {
        data = _mm_add_ps(data, other.data);
        return *this;
    }
    vector4& add(float x, float y, float z) {
        __m128 other = _mm_setr_ps(x, y, z, 0.0f);
        data = _mm_add_ps(data, other);
        return *this;
    }

    vector4& sub(const vector4& other) {
        data = _mm_sub_ps(data, other.data);
        return *this;
    }
    vector4& sub(float x, float y, float z) {
        __m128 other = _mm_setr_ps(x, y, z, 0.0f);
        data = _mm_sub_ps(data, other);
        return *this;
    }

    vector4& mul(float scale) {
        __m128 scalar = _mm_set1_ps(scale);
        data = _mm_mul_ps(data, scalar);
        return *this;
    }
    vector4& mul(float scale, float w_scale) {
        __m128 scalar = _mm_setr_ps(scale, scale, scale, w_scale);
        data = _mm_mul_ps(data, scalar);
        return *this;
    }

    vector4& div(float scale) {
        __m128 scalar = _mm_set1_ps(scale);
        data = _mm_div_ps(data, scalar);
        return *this;
    }
    vector4& div(float scale, float w_scale) {
        __m128 scalar = _mm_setr_ps(scale, scale, scale, w_scale);
        data = _mm_div_ps(data, scalar);
        return *this;
    }

    vector4& dot(const vector4& other) {
        __m128 dp = _mm_dp_ps(data, other.data, 0xFF);
        data = _mm_setr_ps(_mm_cvtss_f32(dp), 0.0f, 0.0f, 0.0f);
        return *this;
    }
    vector4& dot(float x, float y, float z) {
        __m128 other = _mm_setr_ps(x, y, z, 0.0f);
        __m128 dp = _mm_dp_ps(data, other, 0xFF);
        data = _mm_setr_ps(_mm_cvtss_f32(dp), 0.0f, 0.0f, 0.0f);
        return *this;
    }

    float magnitude_square() const {
        __m128 dp = _mm_dp_ps(data, data, 0xFF);
        return _mm_cvtss_f32(dp);
    }
    float magnitude() const {
        return std::sqrt(magnitude_square());
    }

    vector4& normalize() {
        float mag = magnitude();
        if (mag != 0.0f) {
            __m128 scalar = _mm_setr_ps(mag, mag, mag, 1.0f);
            data = _mm_div_ps(data, scalar);
        }
        return *this;
    }
};
