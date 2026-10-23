// Adapted from NVIDIA cuda-samples matrixMul.cu (BSD-3-Clause).
#define TILE 16
extern "C" __global__ void matrix_mul(const half *a, const half *b, half *output) {
  __shared__ half tile_a[TILE][TILE];
  __shared__ half tile_b[TILE][TILE];
  int row = blockIdx.y * TILE + threadIdx.y;
  int col = blockIdx.x * TILE + threadIdx.x;
  float accumulator = 0.0f;
  for (int tile = 0; tile < 64 / TILE; ++tile) {
    tile_a[threadIdx.y][threadIdx.x] = a[row * 64 + tile * TILE + threadIdx.x];
    tile_b[threadIdx.y][threadIdx.x] = b[(tile * TILE + threadIdx.y) * 64 + col];
    __syncthreads();
    for (int k = 0; k < TILE; ++k) accumulator += __half2float(tile_a[threadIdx.y][k]) * __half2float(tile_b[k][threadIdx.x]);
    __syncthreads();
  }
  output[row * 64 + col] = __float2half(accumulator);
}

