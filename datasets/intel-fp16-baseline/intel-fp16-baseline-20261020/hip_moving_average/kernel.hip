// Adapted from ROCm rocm-examples HIP-Basic/moving_average (MIT).
extern "C" __global__ void moving_average(const float *input, float *output, int n, int window) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= n) return;
  int start = max(0, index - window + 1);
  float sum = 0.0f;
  for (int i = start; i <= index; ++i) sum += input[i];
  output[index] = sum / float(index - start + 1);
}

