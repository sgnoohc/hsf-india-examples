#include <cstdlib>
#include <iostream>
#include <random>
#include <chrono>
using namespace std::chrono;

int main(int argc, char** argv)
{
    unsigned long long dim = 2;
    unsigned long long A_nrow = dim;
    unsigned long long A_ncol = dim;
    unsigned long long B_nrow = A_ncol;
    unsigned long long B_ncol = dim;
    unsigned long long C_nrow = A_nrow;
    unsigned long long C_ncol = B_ncol;
    unsigned long long N_A = A_nrow * A_ncol;
    unsigned long long N_B = B_nrow * B_ncol;
    unsigned long long N_C = C_nrow * C_ncol;
    float* A_host = new float[N_A];
    float* B_host = new float[N_B];
    float* C_host = new float[N_C];

    for (unsigned long long ii = 0; ii < N_A; ++ii)
    {
        A_host[ii] = ii;
    }

    for (unsigned long long ii = 0; ii < N_B; ++ii)
    {
        B_host[ii] = 2 * ii;
    }

    for (unsigned long long row = 0; row < C_nrow; ++row)
    {
        for (unsigned long long col = 0; col < C_ncol; ++col)
        {
            for (unsigned long long ii = 0; ii < A_nrow; ++ii)
            {
                C_host[row * C_ncol + col] = A_host[A_ncol * row + ii] * B_host[B_ncol * ii + col];
            }
        }
    }

    for (unsigned long long row = 0; row < C_nrow; ++row)
    {
        for (unsigned long long col = 0; col < C_ncol; ++col)
        {
            std::cout << " " << C_host[row * C_ncol + col];
        }
        std::cout << std::endl;
    }

}
