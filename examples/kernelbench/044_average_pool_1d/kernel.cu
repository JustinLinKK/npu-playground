extern "C" __global__ void kernel(const float *x, float *out) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<396) { int group=i/66, a=i%66; float s=0; for(int u=0;u<8;++u) {int pos=a*1-4+u*1; if(pos>=0 && pos<65) {float v=x[group*65+pos];s+=v;}}out[i]=s/8.0f; }
}
