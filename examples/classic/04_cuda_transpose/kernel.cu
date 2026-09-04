// Adapted from NVIDIA cuda-samples transpose.cu (BSD-3-Clause).
#define TILE_DIM 32
extern "C" __global__ void matrix_transpose(const float *input, float *output) {
  __shared__ float tile[TILE_DIM][TILE_DIM + 1];
  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  if (x < 96 && y < 64) tile[threadIdx.y][threadIdx.x] = input[y * 96 + x];
  __syncthreads();
  x = blockIdx.y * TILE_DIM + threadIdx.x;
  y = blockIdx.x * TILE_DIM + threadIdx.y;
  if (x < 64 && y < 96) output[y * 64 + x] = tile[threadIdx.x][threadIdx.y];
}

