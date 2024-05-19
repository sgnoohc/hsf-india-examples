#include <cstdlib>
#include <iostream>
#include <random>
#include <chrono>
using namespace std::chrono;

//__________________________________________________________________________________________
__global__ void count_darts(double* x, double* y, unsigned long long* counter, unsigned long long N_darts, int i_repeat)
{
    int i_task = threadIdx.x + blockDim.x * blockIdx.x;

    if (i_task >= N_darts)
    {
        return;
    }
    else
    {
        // compute the distance of the dart from the origin
        unsigned long long offset = i_repeat * N_darts;
        double dist = sqrt(x[offset + i_task] * x[offset + i_task] + y[offset + i_task] * y[offset + i_task]);

        printf("i: %d\n", i_task);
        printf("o: %d\n", offset);
        printf("x: %f\n", x[offset + i_task]);
        printf("y: %f\n", y[offset + i_task]);

        // if the distance is less than 1 then count them as inside
        if (dist <= 1)
        {
            // atomic add
            atomicAdd(&counter[i_repeat], 1);
        }
    }
}




//__________________________________________________________________________________________
int main(int argc, char** argv)
{

    //~*~*~*~*~*~*~*~*~*~*~*~*~
    // Option settings
    //~*~*~*~*~*~*~*~*~*~*~*~*~

    int N_repeat = 1;
    unsigned long long N_darts = 1000000; // 1 million random points
    unsigned long long N_thread_per_block = 256; // 256 threads
    bool do_overlap_transfer = false;

    // If arguments are provided overwrite the default setting
    if (argc > 3)
    {
        N_repeat = atoi(argv[1]);
        N_darts = strtoull(argv[2], nullptr, 10);
        N_thread_per_block = strtoull(argv[3], nullptr, 10);
    }
    else if (argc > 2)
    {
        N_repeat = atoi(argv[1]);
        N_darts = strtoull(argv[2], nullptr, 10);
    }
    else if (argc > 1)
    {
        N_repeat = atoi(argv[1]);
    }

    // Starting the clock
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

    //~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
    // The "Answer"
    //~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
    // Create a counter_dart_inside
    // This will count total number of how many fell
    // inside the quarter-circle in all tries
    // Then once we count how many total inside vs. total thrown,
    // from there we can estimate the pi by taking the fraction
    // Since the circle is a unit circle the area is supposed to be pi/4.
    // So the fraction should equal to pi/4.
    unsigned long long counter_dart_inside = 0;


    //~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
    // Creating "darts"
    //~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~

    auto time1 = high_resolution_clock::now();

    unsigned long long N_total_darts = N_darts * N_repeat;

    // create a host (x, y) positions
    double* x_host = new double[N_total_darts];
    double* y_host = new double[N_total_darts];

    for (int i = 0; i < N_repeat; ++i)
    {

        auto time1 = high_resolution_clock::now();

        //~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
        // Create a list of random (x, y)
        //~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~

        // Generate a random (x, y) positions
        for (unsigned int j = 0; j < N_darts; ++j)
        {
            x_host[i * N_darts + j] = distr(gen);
            y_host[i * N_darts + j] = distr(gen);
            std::cout <<  " x_host[i*N_darts+j]: " << x_host[i*N_darts+j] <<  std::endl;
            std::cout <<  " y_host[i*N_darts+j]: " << y_host[i*N_darts+j] <<  std::endl;
            float dist = sqrt(pow(x_host[i*N_darts+j], 2) + pow(y_host[i*N_darts+j], 2));
            std::cout <<  " dist: " << dist <<  std::endl;
        }

    }

    // create a host counter
    unsigned long long* counter_host = new unsigned long long[N_repeat];

    // allocate a device (x, y) positions memory
    double* x_device;
    double* y_device;
    cudaMalloc((void**) &x_device, N_total_darts * sizeof(double));
    cudaMalloc((void**) &y_device, N_total_darts * sizeof(double));

    // allocate a device memory for answers for each repetition
    unsigned long long* counter_device;
    cudaMalloc((void**) &counter_device, N_repeat * sizeof(unsigned long long));

    auto time2 = high_resolution_clock::now();

    //~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
    // Repeating N times to throw more darts
    //~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~

    for (int i = 0; i < N_repeat; ++i)
    {

        // now copy over the host content to the allocated memory space on GPU
        unsigned long long offset = i * N_darts;
        cudaMemcpy(x_device + offset, x_host + offset, N_darts * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(y_device + offset, y_host + offset, N_darts * sizeof(double), cudaMemcpyHostToDevice);

        unsigned long long N_block = (N_darts - 0.5) / N_thread_per_block + 1;
        count_darts<<<N_block, N_thread_per_block>>>(x_device, y_device, counter_device, N_darts, i);

        cudaDeviceSynchronize();

        // Copy back the result
        int counter_offset = i;
        cudaMemcpy(counter_host + counter_offset, counter_device + counter_offset, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

        // Add to the grand counter
        counter_dart_inside += counter_host[i];

    }

    double pi_estimate = ((double)counter_dart_inside) / (N_darts * N_repeat) * 4.;

    std::cout <<  " pi_estimate: " << pi_estimate <<  std::endl;

    free(x_host);
    free(y_host);
    free(counter_host);

    cudaFree(x_device);
    cudaFree(y_device);
    cudaFree(counter_device);

    return 0;

}
