#include <cstdlib>
#include <iostream>
#include <chrono>
using namespace std::chrono;

void vec_add_host(float* A, float* B, float* C, unsigned long long int N_data, unsigned long long int N_ops)
{
    for (unsigned long long int i_data = 0; i_data < N_data; ++i_data)
    {
        for (unsigned i = 0; i < N_ops; ++i)
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
    std::cout << "#            (CPU)              #" << std::endl;
    std::cout << "#                               #" << std::endl;
    std::cout << "#################################" << std::endl;

    unsigned long long int N_data = 10000000;
    unsigned long long int N_ops = 10000;

    auto start = high_resolution_clock::now();

    float* A_host = new float[N_data];
    float* B_host = new float[N_data];
    float* C_host = new float[N_data];

    for (unsigned int i = 0; i < N_data; ++i)
    {
        A_host[i] = i;
        B_host[i] = i * pow(-1, i);
    }

    auto mid = high_resolution_clock::now();

    vec_add_host(A_host, B_host, C_host, N_data, N_ops);

    auto end = high_resolution_clock::now();

    std::cout << " --- Sanity Check ---" << std::endl;
    std::cout << " Printing last 10 result" << std::endl;
    for (unsigned int i = N_data - 10; i < N_data; i++)
    {
        std::cout <<  " i: " << i <<  " C_host[i]: " << C_host[i] <<  std::endl;
    }
    std::cout << std::endl;


    float time_init = duration_cast<microseconds>(mid - start).count() / 1000.;
    float time_exec = duration_cast<microseconds>(end - mid).count() / 1000.;
    float time_tota = duration_cast<microseconds>(end - start).count() / 1000.;

    std::cout <<  " time_init: " << time_init <<  std::endl;
    std::cout <<  " time_exec: " << time_exec <<  std::endl;
    std::cout <<  " time_tota: " << time_tota <<  std::endl;
    std::cout <<  " --- Timing information --- " << std::endl;
    std::cout <<  " time inititalizing       : " << time_init <<  std::endl;
    std::cout <<  " time allocation          : " << "0"       <<  std::endl;
    std::cout <<  " time sending to GPU      : " << "0"       <<  std::endl;
    std::cout <<  " time executing on GPU    : " << time_exec <<  std::endl;
    std::cout <<  " time retrieving from GPU : " << "0"       <<  std::endl;
    std::cout <<  " -------------------------: " <<               std::endl;
    std::cout <<  " time total               : " << time_tota <<  std::endl;

    float P_frac = time_exec / (time_init + time_exec);

    float speedup_ceiling = 1 / (1 - P_frac);

    std::cout <<  " speedup_ceiling: " << speedup_ceiling <<  std::endl;

    free(A_host);
    free(B_host);
    free(C_host);

    return 0;
}
