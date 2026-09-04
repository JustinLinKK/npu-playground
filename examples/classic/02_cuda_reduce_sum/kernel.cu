// Adapted from NVIDIA cuda-samples reduction_kernel.cu (BSD-3-Clause).
extern "C" __global__ void reduce_sum(const float *input, float *output, int n) {
  extern __shared__ float values[];
  unsigned int lane = threadIdx.x;
  unsigned int index = blockIdx.x * blockDim.x * 2 + lane;
  float sum = index < n ? input[index] : 0.0f;
  if (index + blockDim.x < n) sum += input[index + blockDim.x];
  values[lane] = sum;
  __syncthreads();
  for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (lane < stride) values[lane] += values[lane + stride];
    __syncthreads();
  }
  if (lane == 0) output[blockIdx.x] = values[0];
}

