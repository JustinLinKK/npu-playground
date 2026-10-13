extern "C" __global__ void kernel(const float *x, const float *b, float *out) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<1) { float s=0;for(int t=0;t<260;++t) {float d=x[t]-b[t];s+=d*d;}out[i]=s/260.0f; }
}
