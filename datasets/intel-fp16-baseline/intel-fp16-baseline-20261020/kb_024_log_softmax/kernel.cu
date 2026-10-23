extern "C" __global__ void kernel(const float *x, float *out) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<260) { int base=(i/65)*65; float mx=x[base],s=0; for(int t=1;t<65;++t) mx=fmaxf(mx,x[base+t]); for(int t=0;t<65;++t) s+=expf(x[base+t]-mx); out[i]=(x[i]-mx)-logf(s); }
}
