#include <cstdlib>
#include <iostream>
#include <random>
#include <chrono>
using namespace std::chrono;

__global__ void mult(double* A,
                     double* B,
                     double* C,
                     unsigned long long A_nrow,
                     unsigned long long A_ncol,
                     unsigned long long B_nrow,
                     unsigned long long B_ncol,
                     unsigned long long C_nrow,
                     unsigned long long C_ncol,
                     unsigned long long N_ops,
                     bool docoalesced)
{
    unsigned long long row = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned long long col = blockDim.y * blockIdx.y + threadIdx.y;

    // If out of bounds do nothing and return
    if (row >= A_nrow || col >= B_ncol)
        return;

    for (unsigned long long _ = 0; _ < N_ops; ++_)
    {
        for (unsigned long long ii = 0; ii < A_nrow; ++ii)
        {
            if (docoalesced)
                C[row * C_ncol + col] += A[A_ncol * row + ii] * B[B_ncol * col + ii];
            else
                C[row * C_ncol + col] += A[A_ncol * row + ii] * B[B_ncol * ii + col];
        }
    }
}

int main(int argc, char** argv)
{
    if (argc < 4)
    {
        return 1;
    }

    unsigned long long dim = strtoull(argv[1], nullptr, 10);
    unsigned long long N_ops = strtoull(argv[2], nullptr, 10);
    bool docoalesced = atoi(argv[3]);
    unsigned long long A_nrow = 1;
    unsigned long long A_ncol = dim;
    unsigned long long B_nrow = A_ncol;
    unsigned long long B_ncol = 1;
    unsigned long long C_nrow = A_nrow;
    unsigned long long C_ncol = B_ncol;
    unsigned long long N_A = A_nrow * A_ncol;
    unsigned long long N_B = B_nrow * B_ncol;
    unsigned long long N_C = C_nrow * C_ncol;
    double* A_host = new double[N_A];
    double* B_host = new double[N_B];
    double* C_host = new double[N_C];

    for (unsigned long long ii = 0; ii < N_A; ++ii)
    {
        // A_host[ii] = ii;
        A_host[ii] = 1;
    }

    for (unsigned long long ii = 0; ii < N_B; ++ii)
    {
        // B_host[ii] = 2 * ii;
        B_host[ii] = 2;
    }

    double* A_device;
    double* B_device;
    double* C_device;

    cudaMalloc((void**) &A_device, N_A * sizeof(double));
    cudaMalloc((void**) &B_device, N_B * sizeof(double));
    cudaMalloc((void**) &C_device, N_C * sizeof(double));

    cudaMemcpy(A_device, A_host, N_A * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(B_device, B_host, N_B * sizeof(double), cudaMemcpyHostToDevice);

    auto start = high_resolution_clock::now();
    int N_thread_per_block = 16;
    dim3 blockDim(N_thread_per_block, N_thread_per_block, 1);
    dim3 gridDim(int(A_nrow - 0.5)/N_thread_per_block + 1, int(B_ncol - 0.5)/N_thread_per_block + 1, 1);
    mult<<<blockDim, gridDim>>>(A_device, B_device, C_device,
                                A_nrow, A_ncol, B_nrow, B_ncol, C_nrow, C_ncol, N_ops, docoalesced);
    cudaDeviceSynchronize();

    auto end = high_resolution_clock::now();

    double time_tota = duration_cast<microseconds>(end - start).count() / 1000.;

    std::cout <<  " time_tota: " << time_tota <<  std::endl;

    cudaMemcpy(C_host, C_device, N_C * sizeof(double), cudaMemcpyDeviceToHost);

    // for (unsigned long long row = 0; row < C_nrow; ++row)
    // {
    //     for (unsigned long long col = 0; col < C_ncol; ++col)
    //     {
    //         std::cout << " " << C_host[row * C_ncol + col];
    //     }
    //     std::cout << std::endl;
    // }

}
