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
                     MemRef<float, 1> *bias, MemRef<float, 2> *input);

template <typename T, size_t D>
void printVector(MemRef<T, D> &memref, int level = 0) {
  if (level == D - 1) {
      for (int i = 0; i < memref.getSizes()[level]; i++) {
        std::cout << memref.getData()[i] << " ";
      }
  } else {
      for (int i = 0; i < memref.getSizes()[level]; i++) {
        std::cout << "[";
        printVector(memref, level + 1);
        std::cout << "]" << std::endl;
      }
  }
}

int main() {
  /// Initialize data containers.
  MemRef<float, 2> input({4,2});
  MemRef<float, 2> weight({4,2});
  MemRef<float, 1> bias({4});
  MemRef<float, 2> result({4,4});

  int rowNum = 4;
  int colNum = 2;
  for (int i = 0; i < rowNum; i++) {
    for (int j = 0; j < colNum; j++) {
      int index = i * colNum + j;
      input[index] = static_cast<float>(index + 1);
    }
  }

  rowNum = 4;
  colNum = 2;
  for (int i = 0; i < rowNum; i++) {
    for (int j = 0; j < colNum; j++) {
      int index = i * colNum + j;
      weight[index] = static_cast<float>(index + 1);
    }
  }

  rowNum = 1;
  colNum = 4;
  for (int i = 0; i < rowNum; i++) {
    for (int j = 0; j < colNum; j++) {
      int index = i * colNum + j;
      bias[index] = static_cast<float>(0);
      // init result
      result[index] = static_cast<float>(0);
    }
  }

  // Print the generated data to verify
  std::cout << "Input: " << std::endl;
  rowNum = 4;
  colNum = 2;
  for (int i = 0; i < rowNum; i++) {
    for (int j = 0; j < colNum; j++) {
      int index = i * colNum + j;
      std::cout << input[index] << " ";
    }
    std::cout << std::endl;
  }

  std::cout << "Weight: " << std::endl;
  rowNum = 4;
  colNum = 2;
  for (int i = 0; i < rowNum; i++) {
    for (int j = 0; j < colNum; j++) {
      int index = i * colNum + j;
      std::cout << weight[index] << " ";
    }
    std::cout << std::endl;
  }

  std::cout << "Bias: " << std::endl;
  rowNum = 1;
  colNum = 4;
  for (int i = 0; i < rowNum; i++) {
    for (int j = 0; j < colNum; j++) {
      int index = i * colNum + j;
      std::cout << bias[index] << " ";
    }
    std::cout << std::endl;
  }

  const auto inferenceStart = std::chrono::high_resolution_clock::now();

  /// Execute forward inference of the model.
  _mlir_ciface_forward(&result, &weight, &bias, &input);

  const auto inferenceEnd = std::chrono::high_resolution_clock::now();
  const std::chrono::duration<double, std::milli> inferenceTime =
        inferenceEnd - inferenceStart;

  /// Print the output data for verification.
  std::cout << "\033[33;1m[Output] \033[0m";

  std::cout << "Result: " << std::endl;
  rowNum = 4;
  colNum = 4;
  for (int i = 0; i < rowNum; i++) {
    for (int j = 0; j < colNum; j++) {
      int index = i * colNum + j;
      std::cout << result[index] << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  /// Print the performance.
  std::cout << "\033[33;1m[Time] \033[0m";
  std::cout << inferenceTime.count() << " ms"
            << std::endl;
  return 0;
}
