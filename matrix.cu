#include <cstdlib>
#include <iostream>
#include <random>
#include <chrono>
using namespace std::chrono;

int main(int argc, char** argv)
{
    unsigned long long A_nx = 100;
    unsigned long long A_ny = 100;
    unsigned long long B_nx = A_ny;
    unsigned long long B_ny = 100;
    unsigned long long C_nx = A_nx;
    unsigned long long C_ny = B_ny;
    unsigned long long N_A = A_nx * A_ny;
    unsigned long long N_B = B_nx * B_ny;
    unsigned long long N_C = C_nx * C_ny;
    float* A_host = new float[N_A];
    float* B_host = new float[N_B];
    float* C_host = new float[N_C];

    for (unsigned long long ii = 0; ii < N_A; ++ii)
    {
        A_host[ii] = ii;
    }

    for (unsigned long long ii = 0; ii < N_B; ++ii)
    {
        B_host[ii] = ii;
    }

    for (unsigned long long ii = 0; ii < N_B; ++ii)
    {
        B_host[ii] = ii;
    }
}
