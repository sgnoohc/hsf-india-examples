#include <iostream>
#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>

#define myInt_t unsigned long long

__global__ void setup_curandState(curandState* state)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    curand_init(1234, idx, 0, &state[idx]);
}

__global__ void throw_dart(curandState* state, myInt_t* n_inside)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    double x = curand_uniform(&state[idx]);
    double y = curand_uniform(&state[idx]);
    double d = sqrt(x * x + y * y);
    if (d <= 1)
    {
        atomicAdd(n_inside, 1);
    }
}

int main()
{
    //~*~*~*~*~*~*~*~*~*~*~
    // Defining dimensions
    //~*~*~*~*~*~*~*~*~*~*~

    // we will launch 2048 blocks
    myInt_t grid_size = pow(2, 16);

    // we will generate 512 points each block
    myInt_t block_size = 512;

    // total threads
    myInt_t n_total_threads = grid_size * block_size;

    // create a pointer to the array of random state
    // each random state can be used to generate random number
    curandState* state_device;

    // malloc array of random state
    cudaMalloc((void**) &state_device, n_total_threads * sizeof(curandState));

    // actually setup each random state with different index
    setup_curandState<<<grid_size, block_size>>>(state_device);

    // wait until all threads are done
    cudaDeviceSynchronize();

    // setup a counter
    myInt_t* n_inside_device;

    // allocate memory
    cudaMalloc((void**) &n_inside_device, sizeof(myInt_t));

    // actually throw the dart and count how many are inside
    throw_dart<<<grid_size, block_size>>>(state_device, n_inside_device);

    // wait until all threads are done
    cudaDeviceSynchronize();

    // create a counter on host to copy device number to
    myInt_t* n_inside_host = new myInt_t;

    // copy the result to host
    cudaMemcpy(n_inside_host, n_inside_device, sizeof(myInt_t), cudaMemcpyDeviceToHost);

    // estimate pi by counting fraction
    double pi_estimate = (double) *n_inside_host / n_total_threads * 4.;

    // print pi_estimate
    std::cout <<  " pi_estimate: " << pi_estimate <<  std::endl;

    return 0;
}
