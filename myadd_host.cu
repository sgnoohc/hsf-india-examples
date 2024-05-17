#include <cstdlib>
#include <iostream>
#include <chrono>
using namespace std::chrono;

void vec_add_host(float* A, float* B, float* C, unsigned long long int N_data)
{
    for (unsigned long long int i_data = 0; i_data < N_data; ++i_data)
    {
        C[i_data] = A[i_data] + B[i_data];
    }
}

int main(int argc, char** argv)
{

    if (argc < 2)
    {
        std::cout << "Usage:" << std::endl;
        std::cout << std::endl;
        std::cout << "    ./myadd_host_v2 N_data" << std::endl;
        std::cout << std::endl;
        std::cout << std::endl;
    }

    auto start = high_resolution_clock::now();

    unsigned long long int N_data = strtoull(argv[1], nullptr, 10);

    float* A_host = new float[N_data];
    float* B_host = new float[N_data];
    float* C_host = new float[N_data];

    for (unsigned int i = 0; i < N_data; ++i)
    {
        A_host[i] = i;
        B_host[i] = i * pow(-1, i);
    }

    auto mid = high_resolution_clock::now();

    vec_add_host(A_host, B_host, C_host, N_data);

    auto end = high_resolution_clock::now();

    std::cout << "Printing last 10 result" << std::endl;
    for (unsigned int i = N_data - 10; i < N_data; i++)
    {
        std::cout <<  " i: " << i <<  " C_host[i]: " << C_host[i] <<  std::endl;
    }

    float time_init = duration_cast<microseconds>(mid - start).count();
    float time_exec = duration_cast<microseconds>(end - mid).count();

    std::cout <<  " time_init: " << time_init <<  std::endl;
    std::cout <<  " time_exec: " << time_exec <<  std::endl;

    float P_frac = time_exec / (time_init + time_exec);

    float speedup_ceiling = 1 / (1 - P_frac);

    std::cout <<  " speedup_ceiling: " << speedup_ceiling <<  std::endl;

}
