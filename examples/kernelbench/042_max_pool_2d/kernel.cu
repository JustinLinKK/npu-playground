extern "C" __global__ void kernel(const float *x, float *out) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<1058) { int group=i/529, a=(i/23)%23, z=i%23; float s=-3.402823466e38f; for(int u=0;u<4;++u) for(int v=0;v<4;++v) {int r=a*1-1+u*1, c=z*1-1+v*1; if(r>=0 && r<24 && c>=0 && c<24) {float v=x[group*576+r*24+c];s=fmaxf(s,v);}}out[i]=s; }
}
