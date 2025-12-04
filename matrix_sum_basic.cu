#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>

#define MATRIX_DIM 1024

// CUDA kernel: parallel element accumulation using atomic operations
__global__ void accumulateElements(int *input_data, int *output_sum, int total_elements) {
    int row_index = blockIdx.x * blockDim.x + threadIdx.x;
    int col_index = blockIdx.y * blockDim.y + threadIdx.y;
    int linear_index = row_index + col_index * MATRIX_DIM;
    
    if (row_index < MATRIX_DIM && col_index < MATRIX_DIM) {
        atomicAdd(output_sum, input_data[linear_index]);
    }
}

int main() {
    const int element_count = MATRIX_DIM * MATRIX_DIM;
    int *host_input, *host_output;
    int *device_input, *device_output;
    
    // Memory allocation on host
    host_input = (int*)malloc(element_count * sizeof(int));
    host_output = (int*)malloc(sizeof(int));
    *host_output = 0;
    
    // Populate input matrix with test values
    for (int k = 0; k < element_count; k++) {
        host_input[k] = 1;
    }
    
    // Allocate GPU memory
    cudaMalloc((void**)&device_input, element_count * sizeof(int));
    cudaMalloc((void**)&device_output, sizeof(int));
    
    // Transfer data to GPU
    cudaMemcpy(device_input, host_input, element_count * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_output, host_output, sizeof(int), cudaMemcpyHostToDevice);
    
    // Configure and execute kernel
    dim3 thread_config(16, 16);
    dim3 grid_config(MATRIX_DIM / thread_config.x, MATRIX_DIM / thread_config.y);
    accumulateElements<<<grid_config, thread_config>>>(device_input, device_output, element_count);
    
    // Wait for completion and retrieve results
    cudaDeviceSynchronize();
    cudaMemcpy(host_output, device_output, sizeof(int), cudaMemcpyDeviceToHost);
    
    printf("Sum = %d\n", *host_output);
    
    // Release allocated memory
    cudaFree(device_input);
    cudaFree(device_output);
    free(host_input);
    free(host_output);
    
    return 0;
}
