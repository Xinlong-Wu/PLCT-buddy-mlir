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
_mlir_ciface_forward(MemRef<float, 2> *result, MemRef<float, 2> *input1,
                     MemRef<float, 2> *input2, MemRef<float, 2> *bias);

int main() {
  /// Initialize data containers.
  MemRef<float, 2> input1({2, 4});
  MemRef<float, 2> input2({4, 2});
  MemRef<float, 2> bias({4, 4});
  MemRef<float, 2> result({4, 4});

  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 2; j++) {
      int index = i * 2 + j;
      input1[index] = static_cast<float>(index + 1);
    }
  }

  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 4; j++) {
      int index = i * 4 + j;
      input2[index] = static_cast<float>(j + 1);
    }
  }

  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      int index = i * 4 + j;
      bias[index] = static_cast<float>(-1);
    }
  }

  // Print the generated data to verify
  std::cout << "Input1: " << std::endl;
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 2; j++) {
      std::cout << "\t" << input1[i * 2 + j] << " ";
    }
    std::cout << std::endl;
  }

  std::cout << "Input2: " << std::endl;
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 4; j++) {
      std::cout << "\t" << input2[i * 4 + j] << " ";
    }
    std::cout << std::endl;
  }

  std::cout << "Bias: " << std::endl;
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      std::cout << "\t" << bias[i * 4 + j] << " ";
    }
    std::cout << std::endl;
  }

  const auto inferenceStart = std::chrono::high_resolution_clock::now();

  /// Execute forward inference of the model.
  _mlir_ciface_forward(&result, &input1, &input2, &bias);

  const auto inferenceEnd = std::chrono::high_resolution_clock::now();
  const std::chrono::duration<double, std::milli> inferenceTime =
        inferenceEnd - inferenceStart;

  /// Print the output data for verification.
  std::cout << "\033[33;1m[Output] \033[0m";
  std::cout << "[";
  for (int i = 0; i < 4; i++) {
    if (i > 0) std::cout << " ";
    std::cout << "[";
    for (int j = 0; j < 4; j++) {
      if (j > 0) std::cout << " ";
      std::cout << result[i * 4 + j];
    }
    std::cout << "]";
    if (i < 3) std::cout << "\n ";
  }
  std::cout << "]" << std::endl;

  /// Print the performance.
  std::cout << "\033[33;1m[Time] \033[0m";
  std::cout << inferenceTime.count() << " ms"
            << std::endl;
  return 0;
}
