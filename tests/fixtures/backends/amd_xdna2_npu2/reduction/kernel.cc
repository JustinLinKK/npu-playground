extern "C" void compute(float *x0, float *x1, float *x2, float *x3, float *out) {
    float *tiles[] = {x0, x1, x2, x3};
    float sum = 0;
    for (int tile = 0; tile < 4; ++tile)
        for (int i = 0; i < 16; ++i) sum += tiles[tile][i];
    out[0] = sum;
}
