extern "C" __global__ void kernel(const float *x, const float *b, float *out) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<384) { out[i]=x[i/24]*b[i]; }
}
