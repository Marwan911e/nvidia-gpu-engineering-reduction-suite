#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>

#define MATRIX_DIM 1024
#define THREADS_PER_BLOCK 256

// Reduction kernel utilizing shared memory for efficient partial sums
__global__ void computePartialSums(int *input_array, int *partial_results, int array_size) {
    __shared__ int shared_buffer[THREADS_PER_BLOCK];
    
    int local_thread_id = threadIdx.x;
    int global_index = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Transfer data into shared memory
    shared_buffer[local_thread_id] = (global_index < array_size) ? input_array[global_index] : 0;
    __syncthreads();
    
    // Parallel reduction: hierarchical summation
    for (int step_size = blockDim.x / 2; step_size > 0; step_size >>= 1) {
        if (local_thread_id < step_size) {
            shared_buffer[local_thread_id] += shared_buffer[local_thread_id + step_size];
        }
        __syncthreads();
    }
    
    // Store block-level result
    if (local_thread_id == 0) {
        partial_results[blockIdx.x] = shared_buffer[0];
    }
}

int main() {
    const int total_elements = MATRIX_DIM * MATRIX_DIM;
    const int block_count = (total_elements + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    int *host_array, *host_partials;
    int *device_array, *device_partials;
    
    // Host memory setup
    host_array = (int*)malloc(total_elements * sizeof(int));
    host_partials = (int*)malloc(block_count * sizeof(int));
    
    // Initialize test data
    for (int k = 0; k < total_elements; k++) {
        host_array[k] = 1;
    }
    
    // GPU memory allocation
    cudaMalloc((void**)&device_array, total_elements * sizeof(int));
    cudaMalloc((void**)&device_partials, block_count * sizeof(int));
    
    // Host to device transfer
    cudaMemcpy(device_array, host_array, total_elements * sizeof(int), cudaMemcpyHostToDevice);
    
    // Kernel execution configuration
    dim3 grid_layout(block_count, 1, 1);
    dim3 thread_layout(THREADS_PER_BLOCK, 1, 1);
    computePartialSums<<<grid_layout, thread_layout>>>(device_array, device_partials, total_elements);
    
    // Retrieve partial sums
    cudaMemcpy(host_partials, device_partials, block_count * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Final aggregation on CPU
    int final_sum = 0;
    for (int i = 0; i < block_count; i++) {
        final_sum += host_partials[i];
    }
    
    printf("Sum = %d\n", final_sum);
    
    // Cleanup
    cudaFree(device_array);
    cudaFree(device_partials);
    free(host_array);
    free(host_partials);
    
    return 0;
}
