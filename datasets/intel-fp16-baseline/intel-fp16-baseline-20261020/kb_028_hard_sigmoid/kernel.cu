extern "C" __global__ void kernel(const float *x, float *out) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<260) { float v=x[i]; out[i]=fminf(1.0f,fmaxf(0.0f,v/6.0f+0.5f)); }
}
