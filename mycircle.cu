#include <cstdlib>
#include <iostream>
#include <random>
#include <chrono>
using namespace std::chrono;

void help()
{
    std::cout << "Usage:" << std::endl;
    std::cout << std::endl;
    std::cout << "    ./mycircle N_darts" << std::endl;
    std::cout << std::endl;
    std::cout << std::endl;
    return;
}

__global__ void count_darts(double* x, double* y, unsigned long long* counter, int N_darts)
{
    int i_task = threadIdx.x + blockDim.x * blockIdx.x;

    if (i_task >= N_darts)
    {
        return;
    }
    else
    {
        // compute the distance of the dart from the origin
        double dist = sqrt(x[i_task] * x[i_task] + y[i_task] * y[i_task]);

        // if the distance is less than 1 then count them as inside
        if (dist <= 1)
        {
            // atomic add
            atomicAdd(counter, 1);
        }
    }
}

int main(int argc, char** argv)
{

    if (argc > 3)
    {
        unsigned long long N_darts = strtoull(argv[1], nullptr, 10);
    unsigned igned long long N_thread_per_block = strtoull(argv[2], nullptr, 10);
        help();
        return 1;
    }

    auto start = high_resolution_clock::now();

    //~*~*~*~*~*~*~*~*~*~*~*~*~
    // Random Number Generator
    //~*~*~*~*~*~*~*~*~*~*~*~*~

    // create a random device
    std::random_device rd;

    // create a mersenne twistor rng seeded with the random device
    std::mt19937 gen(rd());

    // create a uniform real distribution between [0.0, 1.0)
    std::uniform_real_distribution<> distr(0.0, 1.0);


    //~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
    // Create a list of random (x, y)
    //~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~

    // create a host (x, y) positions
    double* x_host = new double[N_darts];
    double* y_host = new double[N_darts];

    // Generate a random (x, y) positions
    for (unsigned int i = 0; i < N_darts; ++i)
    {
        x_host[i] = distr(gen);
        y_host[i] = distr(gen);
    }

    // create a counter_host
    unsigned long long* counter_host = new unsigned long long;

    // allocate a memory for device GPU
    double* x_device;
    double* y_device;
    cudaMalloc((void**) &x_device, N_darts * sizeof(double));
    cudaMalloc((void**) &y_device, N_darts * sizeof(double));

    // create a counter in device GPU as well
    unsigned long long* counter_device;
    cudaMalloc((void**) &counter_device, sizeof(unsigned long long));

    // now copy over the host content to the allocated memory space on GPU
    cudaMemcpy(x_device, x_host, N_darts * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(y_device, y_host, N_darts * sizeof(double), cudaMemcpyHostToDevice);

    auto mid = high_resolution_clock::now();

    unsigned long long N_block = (N_darts - 0.5) / N_thread_per_block + 1;
    count_darts<<<N_block, N_thread_per_block>>>(x_device, y_device, counter_device, N_darts);

    cudaDeviceSynchronize();

    // Copy back the result
    cudaMemcpy(counter_host, counter_device, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    double pi_estimate = ((double)*counter_host) / N_darts * 4.;

    std::cout <<  " pi_estimate: " << pi_estimate <<  std::endl;

    auto stop = high_resolution_clock::now();

    auto duration_1 = duration_cast<microseconds>(mid - start);
    auto duration_2 = duration_cast<microseconds>(stop - mid);

    std::cout <<  " duration_1.count(): " << duration_1.count() <<  std::endl;
    std::cout <<  " duration_2.count(): " << duration_2.count() <<  std::endl;

    return 0;

}
