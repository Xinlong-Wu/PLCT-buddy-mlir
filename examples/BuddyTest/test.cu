#include <cuda_runtime.h>
#include <cstdio>

#define TASK_ITEM

#ifdef TASK_COLUMN
// Each threade calc one Column
__global__ void addmm_kernel(float* mat1, float* mat2, float* bias, float* result, int m, int n, int k) {
    for (int i = threadIdx.x; i < m; i += blockDim.x) {
        for (int j = threadIdx.y; j < n; j += blockDim.y) {
            printf("BlockDim.x: %d, BlockId.x: %d, TreadId.x: %d, i = %d, j = %d\n", blockDim.x, blockIdx.x, threadIdx.x, i, j);
            float sum = 0.0f;

            // Mat mul
            for (int l = 0; l < k; l++) {
                sum += mat1[i * k + l] * mat2[l * n + j];
            }

            // add bias
            result[i * n + j] = sum + bias[i];
        }
    }
}
#elif defined(TASK_ITEM)
// Each threade calc one Item
__global__ void addmm_kernel(float* mat1, float* mat2, float* bias, float* result, int m, int n, int k) {
    for (int idx = threadIdx.x; idx < m*n; idx += blockDim.x){
        int row = idx / n;
        int col = idx % n;

        if (row < m){
            printf("BlockDim.x: %d, BlockId.x: %d, TreadId.x: %d, row = %d, col = %d\n", blockDim.x, blockIdx.x, threadIdx.x, row, col);
            float sum = 0.0f;

            // Mat mul
            for (int l = 0; l < k; l++) {
                sum += mat1[row * k + l] * mat2[l * n + col];
            }
            result[row * n + col] = sum + bias[row * n + col];
        }
    }
}
#endif

// __global__ void addmm_kernel(float* mat1, float* mat2, float* mat2T, float* bias, float* result, int m, int n, int k) {
//     // transpos 
//     int row = blockIdx.y * blockDim.y + threadIdx.y;
//     int col = blockIdx.x * blockDim.x + threadIdx.x;
//     if (row < k && col < n) {
//         // 将mat2中的元素转置到mat2T中
//         mat2T[col * k + row] = mat2[row * n + col];
//     }

//     // 同步所有线程，确保转置完成
//     __syncthreads();

//     for (int i = threadIdx.x; i < m; i += blockDim.x) {
//         for (int j = threadIdx.y; j < n; j += blockDim.y) {
//             float sum = 0.0f;

//             // Mat mul
//             for (int l = 0; l < k; l++) {
//                 sum += mat1[i * k + l] * mat2[l * n + j];
//             }

//             // add bias
//             result[i * n + j] = sum + bias[i];
//         }
//     }
// }

int main() {
    int m = 4; // row
    int n = 4; // col
    int k = 2; // inner dim

    size_t mat1_size = m * k * sizeof(float);
    size_t mat2_size = k * n * sizeof(float);
    size_t bias_size = m * sizeof(float);
    size_t result_size = m * n * sizeof(float);

    // 初始化矩阵（此处省略初始化代码）
    float h_mat1[4 * 2] = {
        0, 9,
        1, 2,
        3, 4,
        5, 6
    };

    float h_mat2[2 * 4] = {
        1, 2, 3, 4,
        1, 2, 3, 4
    };

    float h_bias[4 * 4] = {
        1, 1, 1, 1,
        1, 1, 1, 1,
        1, 1, 1, 1,
        1, 1, 1, 1
    };

    float h_result[4 * 4] = {-1};


    float* d_mat1;
    float* d_mat2;
    // float* d_mat2T;
    float* d_bias;
    float* d_result;

    cudaMalloc(&d_mat1, mat1_size);
    cudaMalloc(&d_mat2, mat2_size);
    // cudaMalloc(&d_mat2T, mat2_size);
    cudaMalloc(&d_bias, bias_size);
    cudaMalloc(&d_result, result_size);

    cudaMemcpy(d_mat1, h_mat1, mat1_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mat2, h_mat2, mat2_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, h_bias, bias_size, cudaMemcpyHostToDevice);

    dim3 gridSize(1,1);
    dim3 blockSize(2, 1);

    addmm_kernel<<<gridSize, blockSize>>>(d_mat1, d_mat2, d_bias, d_result, m, n, k);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        return -1;
    }

    cudaMemcpy(h_result, d_result, result_size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            printf("%f ", h_result[i * n + j]);
        }
        printf("\n");
    }

    // 清理
    cudaFree(d_mat1);
    cudaFree(d_mat2);
    cudaFree(d_bias);
    cudaFree(d_result);

    return 0;
}
