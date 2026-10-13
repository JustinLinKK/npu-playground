extern "C" __global__ void kernel(const float *x, float *out) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<260) { float s=0;for(int t=i%65;t<65;++t) s+=x[(i/65)*65+t];out[i]=s; }
}
