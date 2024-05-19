#include <cstdlib>
#include <iostream>
#include <random>
#include <chrono>
using namespace std::chrono;

//__________________________________________________________________________________________
__global__ void count_darts(float* x, float* y, unsigned long long* counter, unsigned long long N_darts, int i_repeat, bool verbose)
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
    float xx = x[i_task];
    float yy = y[i_task];
    float dist = sqrt(xx * xx + yy * yy);
    if (verbose)
    {
        printf("i: %llu\n", i_task);
        printf("o: %llu\n", offset);
        printf("x: %f\n", x[i_task]);
        printf("y: %f\n", y[i_task]);
        printf("d: %f\n", dist);
    }

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
    bool verbose = false;

    // Always 256
    unsigned long long N_thread_per_block = 256; // 256 threads

    // If arguments are provided overwrite the default setting
    if (argc > 4)
    {
        N_repeat = atoi(argv[1]);
        N_darts = strtoull(argv[2], nullptr, 10);
        do_overlap_transfer = atoi(argv[3]);
        verbose = atoi(argv[4]);
    }
    else if (argc > 3)
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
    // Creating "darts"
    //~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~

    unsigned long long N_total_darts = N_darts * N_repeat;
    unsigned long long N_block = (N_total_darts - 0.5) / N_thread_per_block + 1;

    // create a host (x, y) positions
    float* x_host = new float[N_total_darts];
    float* y_host = new float[N_total_darts];

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
            if (verbose)
            {
                std::cout <<  " x_host[i*N_darts+j]: " << x_host[i*N_darts+j] <<  std::endl;
                std::cout <<  " y_host[i*N_darts+j]: " << y_host[i*N_darts+j] <<  std::endl;
                std::cout <<  " dist: " << dist <<  std::endl;
            }

        }

    }

    // create a host counter
    unsigned long long* counter_host = new unsigned long long[N_repeat];

    // allocate a device (x, y) positions memory
    float* x_device;
    float* y_device;
    cudaMalloc((void**) &x_device, N_total_darts * sizeof(float));
    cudaMalloc((void**) &y_device, N_total_darts * sizeof(float));

    // allocate a device memory for answers for each repetition
    unsigned long long* counter_device;
    cudaMalloc((void**) &counter_device, N_repeat * sizeof(unsigned long long));

    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    // Create Cuda Stream Lanes
    cudaStream_t stream[N_repeat];

    for (int i = 0; i < N_repeat; ++i)
    {
        cudaStreamCreate(&stream[i]);
    }

    float ms; // elapsed time in milliseconds

    //~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
    // Warmup run
    //~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
    cudaEventRecord(startEvent, 0);
    cudaMemcpy(x_device, x_host, N_total_darts * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(y_device, y_host, N_total_darts * sizeof(float), cudaMemcpyHostToDevice);
    count_darts<<<N_block, N_thread_per_block>>>(x_device, y_device, counter_device, N_total_darts, 0, verbose);
    cudaMemcpy(counter_host, counter_device, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&ms, startEvent, stopEvent);
    printf("Time for sequential transfer and execute (ms): %f\n", ms);

    cudaMemset(counter_device, 0, N_repeat * sizeof(unsigned long long));

    //~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
    // Serial sequence
    //~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~

    auto time_serial_start = high_resolution_clock::now();

    // now copy over the host content to the allocated memory space on GPU
    auto time_tx_start = high_resolution_clock::now();
    cudaEventRecord(startEvent, 0);
    cudaMemcpy(x_device, x_host, N_total_darts * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(y_device, y_host, N_total_darts * sizeof(float), cudaMemcpyHostToDevice);
    auto time_tx_end = high_resolution_clock::now();
    float tx_time = duration_cast<microseconds>(time_tx_end - time_tx_start).count() / 1000.;

    auto time_exec_start = high_resolution_clock::now();
    count_darts<<<N_block, N_thread_per_block>>>(x_device, y_device, counter_device, N_total_darts, 0, verbose);
    cudaDeviceSynchronize();
    auto time_exec_end = high_resolution_clock::now();
    float exec_time = duration_cast<microseconds>(time_exec_end - time_exec_start).count() / 1000.;

    // Copy back the result
    auto time_rx_start = high_resolution_clock::now();
    cudaMemcpy(counter_host, counter_device, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&ms, startEvent, stopEvent);
    printf("Time for sequential transfer and execute (ms): %f\n", ms);
    cudaDeviceSynchronize();
    auto time_rx_end = high_resolution_clock::now();
    float rx_time = duration_cast<microseconds>(time_rx_end - time_rx_start).count() / 1000.;

    auto time_serial_end = high_resolution_clock::now();
    float serial_time = duration_cast<microseconds>(time_serial_end - time_serial_start).count() / 1000.;

    // Compute PI the first counter holds total count
    float pi_estimate_serial = ((float)counter_host[0]) / (N_darts * N_repeat) * 4.;

    std::cout <<  " pi_estimate_serial: " << pi_estimate_serial <<  std::endl;

    std::cout <<  " tx_time: " << tx_time <<  " exec_time: " << exec_time <<  " rx_time: " << rx_time <<  std::endl;
    std::cout <<  " serial_time: " << serial_time <<  std::endl;

    cudaMemset(counter_device, 0, N_repeat * sizeof(unsigned long long));

    //~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
    // Repeating N times to throw more darts
    //~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~

    auto time_overlapping_start = high_resolution_clock::now();

    cudaEventRecord(startEvent, 0);
    for (int i = 0; i < N_repeat; ++i)
    {

        // now copy over the host content to the allocated memory space on GPU
        unsigned long long offset = i * N_darts;
        cudaMemcpyAsync(&x_device[offset], &x_host[offset], N_darts * sizeof(float), cudaMemcpyHostToDevice, stream[i]);
        cudaMemcpyAsync(&y_device[offset], &y_host[offset], N_darts * sizeof(float), cudaMemcpyHostToDevice, stream[i]);

        unsigned long long N_block = (N_darts - 0.5) / N_thread_per_block + 1;
        count_darts<<<N_block, N_thread_per_block, 0, stream[i]>>>(x_device, y_device, counter_device, N_darts, i, verbose);

        // Copy back the result
        int counter_offset = i;
        cudaMemcpyAsync(&counter_host[counter_offset], &counter_device[counter_offset], sizeof(unsigned long long), cudaMemcpyDeviceToHost, stream[i]);
    }
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&ms, startEvent, stopEvent);
    printf("Time for asynchronous transfer and execute (ms): %f\n", ms);

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

    // Add to the grand counter
    for (int i = 0; i < N_repeat; ++i)
    {
        cudaStreamSynchronize(stream[i]);
    }
    auto time_overlapping_end = high_resolution_clock::now();
    float overlapping_time = duration_cast<microseconds>(time_overlapping_end - time_overlapping_start).count() / 1000.;
    // Add to the grand counter
    for (int i = 0; i < N_repeat; ++i)
    {
        counter_dart_inside += counter_host[i];
        // std::cout <<  " counter_dart_inside: " << counter_dart_inside <<  std::endl;
    }

    // Compute PI the first counter holds total count
    float pi_estimate_overlapping = ((float)counter_dart_inside) / (N_darts * N_repeat) * 4.;

    std::cout <<  " pi_estimate_overlapping: " << pi_estimate_overlapping <<  std::endl;

    std::cout <<  " overlapping_time: " << overlapping_time <<  std::endl;

    free(x_host);
    free(y_host);
    free(counter_host);

    cudaFree(x_device);
    cudaFree(y_device);
    cudaFree(counter_device);

    return 0;

}
