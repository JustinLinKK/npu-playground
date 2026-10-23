extern "C" __global__ void kernel(const float *x, float *out) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<8) { int group=i/4, a=(i/2)%2, z=i%2; float s=0; for(int u=0;u<11;++u) for(int v=0;v<11;++v) {int r=a*11-0+u*1, c=z*11-0+v*1; if(r>=0 && r<24 && c>=0 && c<24) {float v=x[group*576+r*24+c];s+=v;}}out[i]=s/121.0f; }
}
