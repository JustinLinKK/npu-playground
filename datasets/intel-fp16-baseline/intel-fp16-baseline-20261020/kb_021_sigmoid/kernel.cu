extern "C" __global__ void kernel(const float *x, float *out) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<260) { float v=x[i]; out[i]=v>=0?1.0f/(1.0f+expf(-v)):expf(v)/(1.0f+expf(v)); }
}
