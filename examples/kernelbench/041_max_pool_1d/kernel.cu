extern "C" __global__ void kernel(const float *x, float *out) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<312) { int group=i/52, a=i%52; float s=-3.402823466e38f; for(int u=0;u<8;++u) {int pos=a*1-4+u*3; if(pos>=0 && pos<65) {float v=x[group*65+pos];s=fmaxf(s,v);}}out[i]=s; }
}
