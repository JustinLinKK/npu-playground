#pragma once
#include <cstdint>

// Portable scalar numerics used identically by Peano and the host compiler.
// Float16 stores IEEE binary16 bits and rounds conversions to nearest, ties to even.
namespace npu {
struct float16 {
    uint16_t bits;
    operator float() const {
        uint32_t sign = uint32_t(bits & 0x8000) << 16;
        uint32_t exponent = (bits >> 10) & 31, fraction = bits & 1023;
        union { uint32_t u; float f; } value;
        if (exponent == 0) {
            if (fraction == 0) value.u = sign;
            else {
                int shift = 0;
                while ((fraction & 1024) == 0) { fraction <<= 1; ++shift; }
                value.u = sign | (uint32_t(113 - shift) << 23) | ((fraction & 1023) << 13);
            }
        } else if (exponent == 31) value.u = sign | 0x7f800000 | (fraction << 13);
        else value.u = sign | ((exponent + 112) << 23) | (fraction << 13);
        return value.f;
    }
    float16 &operator=(float input) {
        union { uint32_t u; float f; } value;
        value.f = input;
        uint32_t sign = (value.u >> 16) & 0x8000;
        int exponent = int((value.u >> 23) & 255) - 112;
        uint32_t fraction = value.u & 0x7fffff;
        if (exponent == 143) { bits = uint16_t(sign | 0x7c00 | (fraction ? 0x200 : 0)); return *this; }
        if (exponent >= 31) { bits = uint16_t(sign | 0x7c00); return *this; }
        if (exponent < -10) { bits = uint16_t(sign); return *this; }
        unsigned shift = exponent <= 0 ? unsigned(14 - exponent) : 13;
        if (exponent <= 0) fraction |= 0x800000;
        uint32_t rounded = fraction >> shift;
        uint32_t remainder = fraction & ((uint32_t(1) << shift) - 1);
        uint32_t halfway = uint32_t(1) << (shift - 1);
        if (remainder > halfway || (remainder == halfway && (rounded & 1))) ++rounded;
        bits = uint16_t(sign | (exponent > 0 ? uint32_t(exponent) << 10 : 0)) + uint16_t(rounded);
        return *this;
    }
};
inline float maximum(float a, float b) { return a > b ? a : b; }
inline float minimum(float a, float b) { return a < b ? a : b; }
inline float absolute(float x) { return x < 0 ? -x : x; }
inline float exp(float x) {
    if (x < -104) return 0;
    int power = int(x * 1.4426950408889634f + (x >= 0 ? .5f : -.5f));
    float r = (x - power * .693145751953125f) - power * 1.428606765330187e-6f;
    float term = 1, value = 1;
    for (int i = 1; i <= 9; ++i) { term *= r / i; value += term; }
    while (power > 0) { value *= 2; --power; }
    while (power < 0) { value *= .5f; ++power; }
    return value;
}
inline float sqrt(float x) {
    if (x == 0) return 0;
    float scale = 1;
    while (x >= 4) { x *= .25f; scale *= 2; }
    while (x < 1) { x *= 4; scale *= .5f; }
    float value = 1.5f;
    for (int i = 0; i < 6; ++i) value = .5f * (value + x / value);
    return value * scale;
}
inline float log(float x) {
    int power = 0;
    while (x >= 2) { x *= .5f; ++power; }
    while (x < 1) { x *= 2; --power; }
    float ratio = (x - 1) / (x + 1), term = ratio, sum = ratio;
    for (int i = 3; i <= 25; i += 2) { term *= ratio * ratio; sum += term / i; }
    return 2 * sum + power * .6931471805599453f;
}
inline float pow(float x, float exponent) { return x == 0 ? 0 : exp(log(x) * exponent); }
}
