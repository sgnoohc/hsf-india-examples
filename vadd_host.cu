#include <cstdlib>
#include <iostream>
#include <chrono>
using namespace std::chrono;

void vec_add_host(float* A, float* B, float* C, unsigned long long int n_data, unsigned long long int n_ops)
{
    for (unsigned long long int i_data = 0; i_data < n_data; ++i_data)
    {
        for (unsigned i = 0; i < n_ops; ++i)
        {
            C[i_data] = A[i_data] + B[i_data];
        }
    }
}

int main(int argc, char** argv)
{

    // banner
    std::cout << "#################################" << std::endl;
    std::cout << "#                               #" << std::endl;
    std::cout << "#    Vector Addition Program    #" << std::endl;
    std::cout << "#            (CPU)              #" << std::endl;
    std::cout << "#                               #" << std::endl;
    std::cout << "#################################" << std::endl;

    // we will have a vector of length n_data
    unsigned long long int n_data = 10000000;

    // and we will sum this up n_ops times
    unsigned long long int n_ops = 1000;

    // print the problem detail
    std::cout << " --- Input data ---" << std::endl;
    std::cout << " n_data = " << n_data << std::endl;
    std::cout << " n_ops  = " << n_ops << std::endl;

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
    auto mid = high_resolution_clock::now();

    // add the vector of length n_data, n_ops time
    vec_add_host(A_host, B_host, C_host, n_data, n_ops);

    // record the current time
    auto end = high_resolution_clock::now();

    // check the result
    std::cout << " --- Sanity Check ---" << std::endl;
    std::cout << " Printing last 10 result" << std::endl;
    for (unsigned int i = n_data - 10; i < n_data; i++)
    {
        std::cout <<  " i: " << i <<  " C_host[i]: " << C_host[i] <<  std::endl;
    }
    std::cout << std::endl;

    // compute the times that it took in each step
    float time_init = duration_cast<microseconds>(mid - start).count() / 1000.;
    float time_exec = duration_cast<microseconds>(end - mid).count() / 1000.;
    float time_tota = duration_cast<microseconds>(end - start).count() / 1000.;

    // print the timing information
    std::cout <<  " --- Timing information --- " << std::endl;
    std::cout <<  " time inititalizing       : " << time_init << " ms" << std::endl;
    std::cout <<  " time executing on CPU    : " << time_exec << " ms" << std::endl;
    std::cout <<  " -------------------------: " <<                       std::endl;
    std::cout <<  " time total               : " << time_tota << " ms" << std::endl;

    // compute the fraction of parallelizable part (the execution part)
    float P_frac = time_exec / (time_init + time_exec);

    // use Amdahl's law to compute the speed up ceiling value
    float speedup_ceiling = 1 / (1 - P_frac);

    // print the ceiling value
    std::cout <<  " speedup_ceiling: " << speedup_ceiling <<  std::endl;

    free(A_host);
    free(B_host);
    free(C_host);

    return 0;
}
