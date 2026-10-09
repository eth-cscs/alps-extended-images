#include <cuda_runtime.h>
#include <cufftdx.hpp>
#include <cmath>

using namespace cufftdx;
using FFT = decltype(Size<16>() + Type<fft_type::c2c>() +
                    Direction<fft_direction::forward>() + Precision<float>() +
                    SM<900>() + Block());

__global__ void fft_kernel(float2* data) {
  using traits = FFT;
  __shared__ typename traits::value_type shared[traits::storage_size];
  typename traits::value_type values[traits::elements_per_thread];
  auto* input = reinterpret_cast<typename traits::value_type*>(data);
  for (unsigned int i = 0; i < traits::elements_per_thread; ++i)
    values[i] = input[threadIdx.x + i * blockDim.x];
  traits().execute(values, shared);
  for (unsigned int i = 0; i < traits::elements_per_thread; ++i)
    input[threadIdx.x + i * blockDim.x] = values[i];
}

int main() {
  float2 host[16]{};
  host[0].x = 1.0f;
  float2* device = nullptr;
  cudaError_t status = cudaMalloc(&device, sizeof(host));
  if (status == cudaSuccess)
    status = cudaMemcpy(device, host, sizeof(host), cudaMemcpyHostToDevice);
  if (status == cudaSuccess) {
    fft_kernel<<<1, FFT::block_dim>>>(device);
    status = cudaDeviceSynchronize();
  }
  if (status == cudaSuccess)
    status = cudaMemcpy(host, device, sizeof(host), cudaMemcpyDeviceToHost);
  if (device) cudaFree(device);
  if (status != cudaSuccess) return 2;
  for (const auto& value : host)
    if (std::fabs(value.x - 1.0f) > 1e-4f || std::fabs(value.y) > 1e-4f)
      return 3;
  return 0;
}
