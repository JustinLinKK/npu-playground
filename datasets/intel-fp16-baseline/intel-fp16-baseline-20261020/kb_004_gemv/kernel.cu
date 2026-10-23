extern "C" __global__ void kernel(const float *x, const float *b, float *out) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<16) { int batch=i/16, r=(i/1)%16, c=i%1; float s=0; for(int t=0;t<32;++t) s+=x[batch*512+(r*32+t)]*b[batch*32+(t*1+c)]; out[i]=s; }
}
