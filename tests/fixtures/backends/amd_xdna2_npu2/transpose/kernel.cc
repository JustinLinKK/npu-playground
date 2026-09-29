extern "C" void compute(float *x, float *out) { for(int i=0;i<8;++i) for(int j=0;j<12;++j) out[j*8+i]=x[i*12+j]; }
