extern "C" __global__ void kernel(const float *x, float *out) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<36) { int base=(i/9)*153+i%9; float s=0; for(int t=0;t<17;++t) {float v=x[base+t*9];s+=v;} out[i]=s/17.0f; }
}
