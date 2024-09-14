//===- test-main.cpp ------------------------------------------------------===//
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//

#include <buddy/Core/Container.h>
#include <buddy/LLM/TextContainer.h>
// #include <filesystem>
#include <chrono>
#include <limits>
#include <string>
#include <utility>
#include <vector>

using namespace buddy;

extern "C" void
_mlir_ciface_forward(MemRef<float, 2> *result, MemRef<float, 2> *weight,
                     MemRef<float, 1> *bias, MemRef<float, 4> *input);

template <typename T, size_t D>
void printVector(MemRef<T, D> &memref, int level = 0) {
  if (level == D - 1) {
      for (int i = 0; i < memref.getSizes()[level]; i++) {
        std::cout << memref.getData()[i] << " ";
      }
      std::cout << std::endl;
  } else {
      for (int i = 0; i < memref.getSizes()[level]; i++) {
        std::cout << "[";
        printVector(memref, level + 1);
        std::cout << "]";
      }
  }
}

int main() {
  /// Initialize data containers.
  MemRef<float, 4> input({1, 1, 16, 16});
  MemRef<float, 2> weight({128, 256});
  MemRef<float, 1> bias({128});
  MemRef<float, 2> result({1, 128});

  int rowNum = 16;
  int colNum = 16;
  for (int i = 0; i < rowNum; i++) {
    for (int j = 0; j < colNum; j++) {
      int index = i * colNum + j;
      input[index] = static_cast<float>(index + 1);
    }
  }

  rowNum = 128;
  colNum = 256;
  for (int i = 0; i < rowNum; i++) {
    for (int j = 0; j < colNum; j++) {
      int index = i * colNum + j;
      weight[index] = static_cast<float>(j + 1);
    }
  }

  rowNum = 1;
  colNum = 128;
  for (int i = 0; i < rowNum; i++) {
    for (int j = 0; j < colNum; j++) {
      int index = i * colNum + j;
      bias[index] = static_cast<float>(-1);
      // init result
      result[index] = static_cast<float>(0);
    }
  }

  // Print the generated data to verify
  std::cout << "Input: " << std::endl;
  printVector(input);

  std::cout << "Weight: " << std::endl;
  printVector(weight);

  std::cout << "Bias: " << std::endl;
  printVector(bias);

  std::cout << "Result: " << std::endl;
  printVector(result);
  std::cout << std::endl;

  const auto inferenceStart = std::chrono::high_resolution_clock::now();

  /// Execute forward inference of the model.
  _mlir_ciface_forward(&result, &weight, &bias, &input);

  const auto inferenceEnd = std::chrono::high_resolution_clock::now();
  const std::chrono::duration<double, std::milli> inferenceTime =
        inferenceEnd - inferenceStart;

  /// Print the output data for verification.
  std::cout << "\033[33;1m[Output] \033[0m";

  std::cout << "Input: " << std::endl;
  printVector(input);

  std::cout << "Weight: " << std::endl;
  printVector(weight);

  std::cout << "Bias: " << std::endl;
  printVector(bias);

  std::cout << "Result: " << std::endl;
  printVector(result);
  std::cout << std::endl;

  /// Print the performance.
  std::cout << "\033[33;1m[Time] \033[0m";
  std::cout << inferenceTime.count() << " ms"
            << std::endl;
  return 0;
}
