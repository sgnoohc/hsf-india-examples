#include <cstdlib>
#include <iostream>
#include <random>
#include <chrono>
using namespace std::chrono;

//__________________________________________________________________________________________
void count_darts_host(float* x, float* y, unsigned long long int* counter, int N_darts)
{
    for (unsigned i_task = 0; i_task < N_darts; ++i_task)
    {
        // compute the distance of the dart from the origin
        float dist = sqrt(x[i_task] * x[i_task] + y[i_task] * y[i_task]);

        // if the distance is less than 1 then count them as inside
        if (dist <= 1)
        {
            // atomic add
            *counter += 1;
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
    if (argc > 2)
    {
        N_repeat = atoi(argv[1]);
        N_darts = strtoull(argv[2], nullptr, 10);
    }
    else if (argc > 1)
    {
        N_darts = strtoull(argv[2], nullptr, 10);
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


    //~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
    // Create a list of random (x, y)
    //~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~

    for (int i = 0; i < N_repeat; ++i)
    {

        // create a host (x, y) positions
        float* x_host = new float[N_darts];
        float* y_host = new float[N_darts];

        // Generate a random (x, y) positions
        for (unsigned int i = 0; i < N_darts; ++i)
        {
            x_host[i] = distr(gen);
            y_host[i] = distr(gen);
        }

        // create a counter_host
        unsigned long long int* counter_host = new unsigned long long int;

        auto mid = high_resolution_clock::now();

        count_darts_host(x_host, y_host, counter_host, N_darts);

        // Add to the grand counter
        counter_dart_inside += *counter_host;

    }

    double pi_estimate = ((double)counter_dart_inside) / (N_darts * N_repeat) * 4.;

    std::cout <<  " pi_estimate: " << pi_estimate <<  std::endl;

    auto stop = high_resolution_clock::now();

    auto duration_1 = duration_cast<microseconds>(mid - start);
    auto duration_2 = duration_cast<microseconds>(stop - mid);

    std::cout <<  " duration_1.count(): " << duration_1.count() <<  std::endl;
    std::cout <<  " duration_2.count(): " << duration_2.count() <<  std::endl;

    return 0;

}
