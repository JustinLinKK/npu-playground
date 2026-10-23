extern "C" __global__ void kernel(const float *x, float *out) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<560) { int base=(i/280)*280+i%35; float s=0; for(int c=0;c<8;++c) {float v=x[base+c*35];s+=v*v;} out[i]=x[i]/sqrtf(s/8+1e-05f); }
}
