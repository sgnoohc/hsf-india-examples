#include <cstdlib>
#include <iostream>
#include <chrono>
using namespace std::chrono;

__global__ void vec_add(const float* A, const float* B, float* C, unsigned long long int n_data, unsigned long long int n_ops)
{

    unsigned long long int i_data = blockDim.x * blockIdx.x + threadIdx.x;
    if (i_data < n_data)
    {
        for (unsigned i = 0; i < n_ops; ++i)
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

    // we will have a vector of length n_data
    unsigned long long int n_data = 10000000;

    // and we will sum this up n_ops times
    unsigned long long int n_ops = 1000;

    // record the current time
    auto start = high_resolution_clock::now();

    // create and allocate memory on the host CPU the vectors
    float* A_host = new float[n_data];
    float* B_host = new float[n_data];
    float* C_host = new float[n_data];

    // set the values of the vectors
    for (unsigned int i = 0; i < n_data; ++i)
    {
        A_host[i] = i;
        B_host[i] = i * pow(-1, i);
    }

    // record the current time
    auto mid1 = high_resolution_clock::now();

    // create pointers and allocate memory on device
    float* A_device;
    float* B_device;
    float* C_device;
    cudaMalloc((void**) &A_device, n_data * sizeof(float));
    cudaMalloc((void**) &B_device, n_data * sizeof(float));
    cudaMalloc((void**) &C_device, n_data * sizeof(float));

    // record the current time
    auto mid2 = high_resolution_clock::now();

    // copy the host input data to the device input data
    cudaMemcpy(A_device, A_host, n_data * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_device, B_host, n_data * sizeof(float), cudaMemcpyHostToDevice);

    // record the current time
    auto mid3 = high_resolution_clock::now();

    // set the block size to some value
    unsigned long long int block_size = 256;

    // then from the block size compute the appropriate grid size
    unsigned long long int grid_size = (n_data - 0.5) / block_size + 1;

    // print the current setup
    std::cout <<  " --- GPU Kernel Launch Config --- " << std::endl;
    std::cout <<  " grid_size: " << grid_size <<  std::endl;
    std::cout <<  " block_size: " << block_size <<  std::endl;
    std::cout << std::endl;

    // add the vector of length n_data, n_ops time
    vec_add<<<grid_size, block_size>>>(A_device, B_device, C_device, n_data, n_ops);

    // wait until all the GPU kernel has finished
    cudaDeviceSynchronize();

    // now we can record the current time once the GPU kernel has all finished
    auto mid4 = high_resolution_clock::now();

    // copy back the result back to the CPU
    cudaMemcpy(C_host, C_device, n_data * sizeof(float), cudaMemcpyDeviceToHost);

    // record the current time where we have finished everything
    auto end = high_resolution_clock::now();

    // perform a sanity check 
    std::cout << " --- Sanity Check ---" << std::endl;
    std::cout << " Printing last 10 result" << std::endl;
    for (unsigned int i = n_data - 10; i < n_data; i++)
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

    // compute the times that it took in each step
    float time_init = duration_cast<microseconds>(mid1 - start).count() / 1000.;
    float time_allo = duration_cast<microseconds>(mid2 - mid1).count() / 1000.;
    float time_send = duration_cast<microseconds>(mid3 - mid2).count() / 1000.;
    float time_exec = duration_cast<microseconds>(mid4 - mid3).count() / 1000.;
    float time_retr = duration_cast<microseconds>(end - mid4).count() / 1000.;
    float time_tota = duration_cast<microseconds>(end - start).count() / 1000.;

    // print the timing information
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
