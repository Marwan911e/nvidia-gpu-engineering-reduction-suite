#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>

#define MATRIX_DIM 1024
#define THREADS_PER_BLOCK 256

// Multi-stage reduction kernel with improved memory access pattern
__global__ void multiStageReduction(int *input_buffer, int *output_buffer, int remaining_count) {
    __shared__ int local_accumulator[THREADS_PER_BLOCK];
    
    int thread_id = threadIdx.x;
    int base_index = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    
    // Each thread processes two elements for better efficiency
    int thread_sum = 0;
    if (base_index < remaining_count) {
        thread_sum = input_buffer[base_index];
    }
    if (base_index + blockDim.x < remaining_count) {
        thread_sum += input_buffer[base_index + blockDim.x];
    }
    
    local_accumulator[thread_id] = thread_sum;
    __syncthreads();
    
    // Binary tree reduction within block
    for (int reduction_step = blockDim.x / 2; reduction_step > 0; reduction_step >>= 1) {
        if (thread_id < reduction_step) {
            local_accumulator[thread_id] += local_accumulator[thread_id + reduction_step];
        }
        __syncthreads();
    }
    
    // Output block result
    if (thread_id == 0) {
        output_buffer[blockIdx.x] = local_accumulator[0];
    }
}

int main() {
    const int total_size = MATRIX_DIM * MATRIX_DIM;
    
    int *host_data;
    int *device_input, *device_output;
    
    // Initialize host data
    host_data = (int*)malloc(total_size * sizeof(int));
    for (int k = 0; k < total_size; k++) {
        host_data[k] = 1;
    }
    
    // Allocate initial GPU buffer
    cudaMalloc((void**)&device_input, total_size * sizeof(int));
    cudaMemcpy(device_input, host_data, total_size * sizeof(int), cudaMemcpyHostToDevice);
    
    int current_count = total_size;
    int block_count;
    
    // Iterative reduction: each kernel launch reduces data size
    while (current_count > 1) {
        block_count = (current_count + (THREADS_PER_BLOCK * 2) - 1) / (THREADS_PER_BLOCK * 2);
        
        cudaMalloc((void**)&device_output, block_count * sizeof(int));
        
        dim3 grid_setup(block_count, 1, 1);
        dim3 thread_setup(THREADS_PER_BLOCK, 1, 1);
        multiStageReduction<<<grid_setup, thread_setup>>>(device_input, device_output, current_count);
        
        cudaDeviceSynchronize();
        
        // Swap buffers for next iteration
        cudaFree(device_input);
        device_input = device_output;
        current_count = block_count;
    }
    
    // Extract final result
    int final_result;
    cudaMemcpy(&final_result, device_input, sizeof(int), cudaMemcpyDeviceToHost);
    
    printf("Sum = %d\n", final_result);
    
    // Memory cleanup
    cudaFree(device_input);
    free(host_data);
    
    return 0;
}
