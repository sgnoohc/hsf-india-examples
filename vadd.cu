#include <cstdlib>
#include <iostream>
#include <chrono>
using namespace std::chrono;

__global__ void vec_add(const float* A, const float* B, float* C, unsigned long long int N_data, unsigned long long int N_ops)
{

    unsigned long long int i_data = blockDim.x * blockIdx.x + threadIdx.x;
    if (i_data < N_data)
    {
        for (unsigned i = 0; i < N_ops; ++i)
        {
            C[i_data] = A[i_data] + B[i_data];
        }
    }
}

int main(int argc, char** argv)
{

    std::cout << "#################################" << std::endl;
    std::cout << "#                               #" << std::endl;
    std::cout << "#    Vector Addition Program    #" << std::endl;
    std::cout << "#            (GPU)              #" << std::endl;
    std::cout << "#                               #" << std::endl;
    std::cout << "#################################" << std::endl;

    unsigned long long int N_data = 10000000;
    unsigned long long int N_ops = 1000;

    auto start = high_resolution_clock::now();

    float* A_host = new float[N_data];
    float* B_host = new float[N_data];
    float* C_host = new float[N_data];

    for (unsigned int i = 0; i < N_data; ++i)
    {
        A_host[i] = i;
        B_host[i] = i * pow(-1, i);
    }

    auto mid1 = high_resolution_clock::now();

    // allocate memory on device
    float* A_device;
    float* B_device;
    float* C_device;
    cudaMalloc((void**) &A_device, N_data * sizeof(float));
    cudaMalloc((void**) &B_device, N_data * sizeof(float));
    cudaMalloc((void**) &C_device, N_data * sizeof(float));

    auto mid2 = high_resolution_clock::now();

    // copy the host input data to the device input data
    cudaMemcpy(A_device, A_host, N_data * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_device, B_host, N_data * sizeof(float), cudaMemcpyHostToDevice);

    auto mid3 = high_resolution_clock::now();

    unsigned long long int block_size = 256;
    unsigned long long int grid_size = (N_data - 0.5) / block_size + 1;

    std::cout <<  " --- GPU Kernel Launch Config --- " << std::endl;
    std::cout <<  " grid_size: " << grid_size <<  std::endl;
    std::cout <<  " block_size: " << block_size <<  std::endl;
    std::cout << std::endl;

    vec_add<<<grid_size, block_size>>>(A_device, B_device, C_device, N_data, N_ops);
    cudaDeviceSynchronize();

    auto mid4 = high_resolution_clock::now();

    cudaMemcpy(C_host, C_device, N_data * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << " --- Sanity Check ---" << std::endl;
    std::cout << " Printing last 10 result" << std::endl;
    for (unsigned int i = N_data - 10; i < N_data; i++)
    {
        std::cout <<  " i: " << i <<  " C_host[i]: " << C_host[i] <<  std::endl;
    }
    std::cout << std::endl;

    // Free allocated memory
    free(A_host);
    free(B_host);
    free(C_host);
    cudaFree(A_device);
    cudaFree(B_device);
    cudaFree(C_device);

    auto end = high_resolution_clock::now();

    float time_init = duration_cast<microseconds>(mid1 - start).count() / 1000.;
    float time_allo = duration_cast<microseconds>(mid2 - mid1).count() / 1000.;
    float time_send = duration_cast<microseconds>(mid3 - mid2).count() / 1000.;
    float time_exec = duration_cast<microseconds>(mid4 - mid3).count() / 1000.;
    float time_retr = duration_cast<microseconds>(end - mid4).count() / 1000.;
    float time_tota = duration_cast<microseconds>(end - start).count() / 1000.;

    std::cout <<  " --- Timing information --- " << std::endl;
    std::cout <<  " time inititalizing       : " << time_init << " ms" <<  std::endl;
    std::cout <<  " time allocation          : " << time_allo << " ms" <<  std::endl;
    std::cout <<  " time sending to GPU      : " << time_send << " ms" <<  std::endl;
    std::cout <<  " time executing on GPU    : " << time_exec << " ms" <<  std::endl;
    std::cout <<  " time retrieving from GPU : " << time_retr << " ms" <<  std::endl;
    std::cout <<  " -------------------------: " <<                        std::endl;
    std::cout <<  " time total               : " << time_tota << " ms" <<  std::endl;

    return 0;
}
