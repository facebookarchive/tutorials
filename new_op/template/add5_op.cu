/**
Copyright (c) Facebook, Inc. and its affiliates.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include "caffe2/operators/add5_op.h"

#include "caffe2/core/context_gpu.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <typename T>
__global__ void Add5Kernel(const int N, const T* data, T* output) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    // TODO - 3
  }
}

template <>
template <typename T>
bool Add5Op<CUDAContext>::DoRunWithType() {
  const auto& data = Input(DATA);
  const auto N = data.size();
  const auto* data_ptr = data.template data<T>();
  auto* output = Output(0);
  output->ResizeLike(data);
  auto* output_ptr = output->template mutable_data<T>();

  Add5Kernel<<<
      CAFFE_GET_BLOCKS(N),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(N, data_ptr, output_ptr);
  return true;
}

template <typename T>
__global__ void Add5GradientKernel(const int N, const T* data, T* output) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    // GI[0] = GO[0]
    // TODO - 4
  }
}

template <>
template <typename T>
bool Add5GradientOp<CUDAContext>::DoRunWithType() {
  const auto& data = Input(DATA);
  const auto N = data.size();
  const auto* data_ptr = data.template data<T>();
  auto* output = Output(0);
  output->ResizeLike(data);
  auto* output_ptr = output->template mutable_data<T>();

  Add5GradientKernel<<<
      CAFFE_GET_BLOCKS(N),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(N, data_ptr, output_ptr);
  return true;
}

REGISTER_CUDA_OPERATOR(Add5, Add5Op<CUDAContext>);
REGISTER_CUDA_OPERATOR(Add5Gradient, Add5GradientOp<CUDAContext>);

} // namespace caffe2
