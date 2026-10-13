extern "C" __global__ void kernel(const float *x, const float *b, float *out) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<384) { int batch=i/384, r=(i/24)%16, c=i%24; float s=0; for(int t=0;t<32;++t) s+=x[batch*512+(t*16+r)]*b[batch*768+(t*24+c)]; out[i]=s; }
}
