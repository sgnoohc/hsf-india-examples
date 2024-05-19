#include <cstdlib>
#include <iostream>
#include <random>
#include <chrono>
using namespace std::chrono;

//__________________________________________________________________________________________
__global__ void count_darts(double* x, double* y, unsigned long long* counter, unsigned long long N_darts, int i_repeat)
{
    // Get this specific thread unique index
    unsigned long long i_task = threadIdx.x + blockDim.x * blockIdx.x;

    // check that it is within N_darts if not exit
    if (i_task >= N_darts)
        return;

    // compute offset and correct i_task
    unsigned long long offset = i_repeat * N_darts;
    i_task += offset;

    // compute the distance of the dart from the origin
    double dist = sqrt(x[i_task] * x[i_task] + y[i_task] * y[i_task]);
    // printf("i: %llu\n", i_task);
    // printf("o: %llu\n", offset);
    // printf("x: %f\n", x[offset + i_task]);
    // printf("y: %f\n", y[offset + i_task]);
    // printf("d: %f\n", dist);

    // if the distance is less than 1 then count them as inside
    if (dist <= 1)
    {
        // atomic add
        atomicAdd(&counter[i_repeat], 1);
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
    bool do_overlap_transfer = false;

    // Always 256
    unsigned long long N_thread_per_block = 256; // 256 threads

    // If arguments are provided overwrite the default setting
    if (argc > 3)
    {
        N_repeat = atoi(argv[1]);
        N_darts = strtoull(argv[2], nullptr, 10);
        do_overlap_transfer = atoi(argv[3]);
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

    unsigned long long N_total_darts = N_darts * N_repeat;

    // create a host (x, y) positions
    double* x_host = new double[N_total_darts];
    double* y_host = new double[N_total_darts];

    for (int i = 0; i < N_repeat; ++i)
    {

        //~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
        // Create a list of random (x, y)
        //~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~

        // Generate a random (x, y) positions
        for (unsigned int j = 0; j < N_darts; ++j)
        {
            x_host[i * N_darts + j] = distr(gen);
            y_host[i * N_darts + j] = distr(gen);
            float dist = sqrt(pow(x_host[i*N_darts+j], 2) + pow(y_host[i*N_darts+j], 2));
            // std::cout <<  " x_host[i*N_darts+j]: " << x_host[i*N_darts+j] <<  std::endl;
            // std::cout <<  " y_host[i*N_darts+j]: " << y_host[i*N_darts+j] <<  std::endl;
            // std::cout <<  " dist: " << dist <<  std::endl;

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

    //~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
    // Repeating N times to throw more darts
    //~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~

    // Starting the clock
    auto start = high_resolution_clock::now();

    if (do_overlap_transfer)
    {
        // Create Cuda Stream Lanes
        cudaStream_t stream[N_repeat];

        for (int i = 0; i < N_repeat; ++i)
        {
            cudaStreamCreate(&stream[i]);
        }

        for (int i = 0; i < N_repeat; ++i)
        {

            // now copy over the host content to the allocated memory space on GPU
            unsigned long long offset = i * N_darts;
            cudaMemcpyAsync(&x_device[offset], &x_host[offset], N_darts * sizeof(double), cudaMemcpyHostToDevice, stream[i]);
            cudaMemcpyAsync(&y_device[offset], &y_host[offset], N_darts * sizeof(double), cudaMemcpyHostToDevice, stream[i]);

            unsigned long long N_block = (N_darts - 0.5) / N_thread_per_block + 1;
            count_darts<<<N_block, N_thread_per_block, 0, stream[i]>>>(x_device, y_device, counter_device, N_darts, i);

            // Copy back the result
            int counter_offset = i;
            cudaMemcpyAsync(&counter_host[counter_offset], &counter_device[counter_offset], sizeof(unsigned long long), cudaMemcpyDeviceToHost, stream[i]);
        }

        // Add to the grand counter
        for (int i = 0; i < N_repeat; ++i)
        {
            cudaStreamSynchronize(stream[i]);
            counter_dart_inside += counter_host[i];
            // std::cout <<  " counter_dart_inside: " << counter_dart_inside <<  std::endl;
        }

    }
    else
    {
        for (int i = 0; i < N_repeat; ++i)
        {

            // now copy over the host content to the allocated memory space on GPU
            auto time_tx_start = high_resolution_clock::now();
            unsigned long long offset = i * N_darts;
            cudaMemcpy(&x_device[offset], &x_host[offset], N_darts * sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(&y_device[offset], &y_host[offset], N_darts * sizeof(double), cudaMemcpyHostToDevice);
            auto time_tx_end = high_resolution_clock::now();
            float tx_time = duration_cast<microseconds>(time_tx_end - time_tx_start).count() / 1000.;

            auto time_exec_start = high_resolution_clock::now();
            unsigned long long N_block = (N_darts - 0.5) / N_thread_per_block + 1;
            count_darts<<<N_block, N_thread_per_block>>>(x_device, y_device, counter_device, N_darts, i);
            auto time_exec_end = high_resolution_clock::now();
            float exec_time = duration_cast<microseconds>(time_exec_end - time_exec_start).count() / 1000.;

            // Copy back the result
            auto time_rx_start = high_resolution_clock::now();
            int counter_offset = i;
            cudaMemcpy(&counter_host[counter_offset], &counter_device[counter_offset], sizeof(unsigned long long), cudaMemcpyDeviceToHost);
            auto time_rx_end = high_resolution_clock::now();
            float rx_time = duration_cast<microseconds>(time_rx_end - time_rx_start).count() / 1000.;

            // Add to the grand counter
            counter_dart_inside += counter_host[i];

            // std::cout <<  " counter_dart_inside: " << counter_dart_inside <<  std::endl;

            std::cout <<  " i: " << i <<  " tx_time: " << tx_time <<  " exec_time: " << exec_time <<  " rx_time: " << rx_time <<  std::endl;

        }
    }

    // Starting the clock
    auto end = high_resolution_clock::now();

    float time = duration_cast<microseconds>(end - start).count() / 1000.;

    double pi_estimate = ((double)counter_dart_inside) / (N_darts * N_repeat) * 4.;

    std::cout <<  " pi_estimate: " << pi_estimate <<  std::endl;
    std::cout <<  " time: " << time <<  std::endl;

    free(x_host);
    free(y_host);
    free(counter_host);

    cudaFree(x_device);
    cudaFree(y_device);
    cudaFree(counter_device);

    return 0;

}
