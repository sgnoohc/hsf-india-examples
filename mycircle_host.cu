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

int main(int argc, char** argv)
{

    if (argc < 3)
    {
        help();
        return 1;
    }

    auto start = high_resolution_clock::now();
    unsigned long long int N_darts = strtoull(argv[1], nullptr, 10);
    unsigned long long int N_thread_per_block = strtoull(argv[2], nullptr, 10);

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

    double pi_estimate = ((double)*counter_host) / N_darts * 4.;

    std::cout <<  " pi_estimate: " << pi_estimate <<  std::endl;

    auto stop = high_resolution_clock::now();

    auto duration_1 = duration_cast<microseconds>(mid - start);
    auto duration_2 = duration_cast<microseconds>(stop - mid);

    std::cout <<  " duration_1.count(): " << duration_1.count() <<  std::endl;
    std::cout <<  " duration_2.count(): " << duration_2.count() <<  std::endl;

    return 0;

}
