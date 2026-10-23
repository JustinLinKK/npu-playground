extern "C" __global__ void kernel(const float *x, float *out) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<260) { float v=x[i]; out[i]=v>20?v:log1pf(expf(v)); }
}
