#include <cstdlib>
#include <iostream>
#include <random>
#include <chrono>
using namespace std::chrono;

__global__ void mult(float* A,
                     float* B,
                     float* C,
                     unsigned long long A_nrow,
                     unsigned long long A_ncol,
                     unsigned long long B_nrow,
                     unsigned long long B_ncol,
                     unsigned long long C_nrow,
                     unsigned long long C_ncol)
{
    unsigned long long row = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned long long col = blockDim.y * blockIdx.y + threadIdx.y;

    // If out of bounds do nothing and return
    if (row >= A_nrow || col >= B_ncol)
        return;

    for (unsigned long long ii = 0; ii < A_nrow; ++ii)
    {
        C[row * C_ncol + col] = A[A_ncol * row + ii] * B[B_ncol * ii + col];
    }
}

int main(int argc, char** argv)
{
    unsigned long long dim = 20;
    unsigned long long A_nrow = dim;
    unsigned long long A_ncol = dim;
    unsigned long long B_nrow = A_ncol;
    unsigned long long B_ncol = dim;
    unsigned long long C_nrow = A_nrow;
    unsigned long long C_ncol = B_ncol;
    unsigned long long N_A = A_nrow * A_ncol;
    unsigned long long N_B = B_nrow * B_ncol;
    unsigned long long N_C = C_nrow * C_ncol;
    float* A_host = new float[N_A];
    float* B_host = new float[N_B];
    float* C_host = new float[N_C];

    for (unsigned long long ii = 0; ii < N_A; ++ii)
    {
        A_host[ii] = ii;
    }

    for (unsigned long long ii = 0; ii < N_B; ++ii)
    {
        B_host[ii] = 2 * ii;
    }

    float* A_device;
    float* B_device;
    float* C_device;

    cudaMalloc((void**) &A_device, N_A * sizeof(float));
    cudaMalloc((void**) &B_device, N_B * sizeof(float));
    cudaMalloc((void**) &C_device, N_C * sizeof(float));

    cudaMemcpy(A_device, A_host, N_A * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_device, B_host, N_B * sizeof(float), cudaMemcpyHostToDevice);

    N_thread_per_block = 16;
    dim3 blockDim(N_thread_per_block, N_thread_per_block, 1);
    dim3 gridDim(int(A_nrow - 0.5)/N_thread_per_block + 1, int(B_ncol - 0.5)/N_thread_per_block + 1, 1);
    mult<<<blockDim, gridDim>>>(A_device, B_device, C_device,
                                A_nrow, A_ncol, B_nrow, B_ncol, C_nrow, C_ncol);
    cudaDeviceSynchronize();

    cudaMemcpy(C_host, C_device, N_C * sizeof(float), cudaMemcpyDeviceToHost);

    for (unsigned long long row = 0; row < C_nrow; ++row)
    {
        for (unsigned long long col = 0; col < C_ncol; ++col)
        {
            std::cout << " " << C_host[row * C_ncol + col];
        }
        std::cout << std::endl;
    }

}
