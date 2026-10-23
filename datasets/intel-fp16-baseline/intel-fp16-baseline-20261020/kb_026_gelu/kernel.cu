extern "C" __global__ void kernel(const float *x, float *out) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<260) { float v=x[i]; out[i]=0.5f*v*(1.0f+erff(v*0.7071067811865475f)); }
}
