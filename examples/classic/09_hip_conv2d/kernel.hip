// Adapted from ROCm rocm-examples image_convolution (MIT).
extern "C" __global__ void image_convolution(const float *input, const float *weight, float *output) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= 32 || col >= 32) return;
  float sum = 0.0f;
  for (int ky = -1; ky <= 1; ++ky) {
    for (int kx = -1; kx <= 1; ++kx) {
      int y = row + ky;
      int x = col + kx;
      if (y >= 0 && y < 32 && x >= 0 && x < 32) sum += input[y * 32 + x] * weight[(ky + 1) * 3 + kx + 1];
    }
  }
  output[row * 32 + col] = sum;
}

