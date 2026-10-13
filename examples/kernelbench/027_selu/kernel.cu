extern "C" __global__ void kernel(const float *x, float *out) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<260) { float v=x[i]; out[i]=1.0507009873554805f*(v>0?v:1.6732632423543772f*expm1f(v)); }
}
