#include <stdio.h>


__global__ void vector_add(float *A, float *B, float *C, int numElements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        C[i] = A[i] + B[i];
    }
}

__global__ void scalar_multiply(float *C, float *D, float scalar, int numElements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        D[i] = C[i] * scalar;
    }
}

__global__ void fused_operation(float *A, float *B, float *D, float scalar, int numElements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        float C = A[i] + B[i];
        D[i] = C * scalar;
    }
}

int main() {
    int numElements = 50000;
    size_t size = numElements * sizeof(float);
    float scalarValue = 2.0;

    // Memory allocation and initialization code
    float *h_A = new float[numElements];
    float *h_B = new float[numElements];
    for (int i = 0; i < numElements; i++) {
        h_A[i] = rand() / static_cast<float>(RAND_MAX);
        h_B[i] = rand() / static_cast<float>(RAND_MAX);
    }

    float *d_A, *d_B, *d_C, *d_D;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);
    cudaMalloc((void**)&d_D, size);
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    const int threadsPerBlock = 256;  
    const int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

    // Non-fused version timing
    float milliseconds1 = 0;
    cudaEvent_t start1, stop1;
    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);

    cudaEventRecord(start1);
    vector_add<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
    scalar_multiply<<<blocksPerGrid, threadsPerBlock>>>(d_C, d_D, scalarValue, numElements);
    cudaEventRecord(stop1);
    cudaEventSynchronize(stop1);
    cudaEventElapsedTime(&milliseconds1, start1, stop1);

    // Fused version timing
    float milliseconds2 = 0;
    cudaEvent_t start2, stop2;
    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);
    cudaEventRecord(start2);
    fused_operation<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_D, scalarValue, numElements);
    cudaEventRecord(stop2);
    cudaEventSynchronize(stop2);
    cudaEventElapsedTime(&milliseconds2, start2, stop2);

    // Allocate memory for host results
    float *h_Result_NonFused = new float[numElements];
    float *h_Result_Fused = new float[numElements];

    // Copy results from device to host for non-fused operation
    cudaMemcpy(h_Result_NonFused, d_D, size, cudaMemcpyDeviceToHost);

    // Execute fused operation
    fused_operation<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_D, scalarValue, numElements);

    // Copy results from device to host for fused operation
    cudaMemcpy(h_Result_Fused, d_D, size, cudaMemcpyDeviceToHost);

    // Compare and print results
    bool resultsMatch = true;
    for (int i = 0; i < numElements; i++) {
        printf("Element %d: Non-Fused = %f, Fused = %f\n", i, h_Result_NonFused[i], h_Result_Fused[i]);
        if (h_Result_NonFused[i] != h_Result_Fused[i]) {
            resultsMatch = false;
        }
    }

    if (resultsMatch) {
        printf("Results match!\n");
    } else {
        printf("Results differ!\n");
    }

    printf("Time taken by non-fused version: %f ms\n", milliseconds1);
    printf("Time taken by fused version: %f ms\n", milliseconds2);




    // Cleanup code
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_D);
    cudaEventDestroy(start1);
    cudaEventDestroy(stop1);
    cudaEventDestroy(start2);
    cudaEventDestroy(stop2);

    return 0;
}
