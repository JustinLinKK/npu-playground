// Adapted from ROCm's basic image gamma-correction tutorial (MIT).
extern "C" __global__ void gamma_correction(const float *input, float *output, int n, float gamma) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
    float value = fminf(1.0f, fmaxf(0.0f, input[index]));
    output[index] = powf(value, gamma);
  }
}

