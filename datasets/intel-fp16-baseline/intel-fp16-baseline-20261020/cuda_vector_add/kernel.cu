// Adapted from NVIDIA cuda-samples vectorAdd.cu (BSD-3-Clause).
extern "C" __global__ void vector_add(const float *x, const float *y, float *output, int n) {
  int index = blockDim.x * blockIdx.x + threadIdx.x;
  if (index < n) output[index] = x[index] + y[index];
}

