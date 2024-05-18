#include <cstdlib>
#include <iostream>
#include <chrono>
using namespace std::chrono;

__global__ void vec_add(const float* A, const float* B, float* C, unsigned long long int N_data)
{

    unsigned long long int i_data = blockDim.x * blockIdx.x + threadIdx.x;
    if (i_data < N_data)
    {
        for (unsigned i = 0; i < 200; ++i)
        {
            C[i_data] = A[i_data] + B[i_data];
        }
    }
}

int main(int argc, char** argv)
{

    if (argc < 2)
    {
        std::cout << "Usage:" << std::endl;
        std::cout << std::endl;
        std::cout << "    " << argv[0] << " N_data" << std::endl;
        std::cout << std::endl;
        std::cout << std::endl;
        return 1;
    }

    auto start = high_resolution_clock::now();

    unsigned long long int N_data = strtoull(argv[1], nullptr, 10);

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

    unsigned long long int N_thread_per_block = 256;
    unsigned long long int N_block = (N_data - 0.5) / N_thread_per_block + 1;

    std::cout <<  " N_block: " << N_block <<  std::endl;
    std::cout <<  " N_thread_per_block: " << N_thread_per_block <<  std::endl;

    vec_add<<<N_block, N_thread_per_block>>>(A_device, B_device, C_device, N_data);

    auto mid4 = high_resolution_clock::now();

    cudaMemcpy(C_host, C_device, N_data * sizeof(float), cudaMemcpyDeviceToHost);

    auto end = high_resolution_clock::now();

    std::cout << "Printing last 10 result" << std::endl;
    for (unsigned int i = N_data - 10; i < N_data; i++)
    {
        std::cout <<  " i: " << i <<  " C_host[i]: " << C_host[i] <<  std::endl;
    }

    float time_init = duration_cast<microseconds>(mid1 - start).count();
    float time_allo = duration_cast<microseconds>(mid2 - mid1).count();
    float time_send = duration_cast<microseconds>(mid3 - mid2).count();
    float time_exec = duration_cast<microseconds>(mid4 - mid3).count();
    float time_retr = duration_cast<microseconds>(end - mid4).count();

    std::cout <<  " time_init: " << time_init <<  std::endl;
    std::cout <<  " time_allo: " << time_allo <<  std::endl;
    std::cout <<  " time_send: " << time_send <<  std::endl;
    std::cout <<  " time_exec: " << time_exec <<  std::endl;
    std::cout <<  " time_retr: " << time_retr <<  std::endl;

    // Free allocated memory
    free(A_host);
    free(B_host);
    free(C_host);
    cudaFree(A_device);
    cudaFree(B_device);
    cudaFree(C_device);

    return 0;
}
