#include <cstdint>
extern "C" void add(float *x, float *y, float *output, int32_t n) {
    for (int32_t i = 0; i < n; ++i) output[i] = x[i] + y[i];
}
extern "C" void copy(float *x, float *out, int32_t n) { for(int32_t i=0;i<n;++i) out[i]=x[i]; }
