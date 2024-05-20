#include <cstdlib>
#include <iostream>
#include <random>
#include <chrono>
using namespace std::chrono;

__global__ void madd(float* A,
                     float* B,
                     int m_dim,
                     int ioffset)
{
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    int col = blockDim.y * blockIdx.y + threadIdx.y;

    // If out of bounds do nothing and return
    if (row >= m_dim || col >= m_dim)
        return;

    int idx = m_dim * row + col + ioffset * m_dim * m_dim;
    float Aval = A[idx];
    B[idx] = Aval + Aval;
}

int main(int argc, char** argv)
{

    std::cout << "################################" << std::endl;
    std::cout << "#                              #" << std::endl;
    std::cout << "#         Matrix Sum           #" << std::endl;
    std::cout << "#     (Overlap Transfer)       #" << std::endl;
    std::cout << "#                              #" << std::endl;
    std::cout << "################################" << std::endl;

    // goal: Perform N times A + A = B matrix squaring
    const int n_repeat = 4;

    // A matrix dimension and total element definition
    const int m_dim = 2048;
    const int m_tot = m_dim * m_dim;

    // we will perform each element as one thread
    int block_len = 16;

    // then the block dimensions are defined
    dim3 block_size(block_len, block_len, 1);

    // compute number of blocks in each dimension
    int grid_len = int(m_dim - 0.5) / block_len + 1;

    // then for grid size needs to be computed to cover the entire elements
    dim3 grid_size(grid_len, grid_len, 1);

    // define cuda events
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    // // create cuda streams
    // cudaStream_t stream[n_repeat];
    // for (int i = 0; i < n_repeat; ++i)
    // {
    //     cudaStreamCreate(&stream[i]);
    // }

    // variable to read out time
    float ms;

    // we store in single dimension where row is assumed first
    float* A_host = new float[m_tot];
    float* B_host = new float[m_tot];

    // for now for simplicity we set all to 1 for A
    for (int ii = 0; ii < m_tot; ++ii)
    {
        A_host[ii] = 1;
    }

    //~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*
    //~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*
    //~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*

    printf(" --- Sequential Run ---\n");

    // create pointer to the device matrix input
    float* A_device;
    float* B_device;

    // allocate memory to the pointer
    cudaMalloc((void**) &A_device, m_tot * sizeof(float));
    cudaMalloc((void**) &B_device, m_tot * sizeof(float));

    // warm up run
    madd<<<grid_size, block_size>>>( A_device, B_device, m_dim, 0 /*no offset*/);

    // start the overall timer
    cudaEventRecord(startEvent, 0);

    for (int i = 0 ; i < n_repeat; ++i)
    {
        // copy the host values to device memory
        cudaMemcpy(A_device, A_host, m_tot * sizeof(float), cudaMemcpyHostToDevice);

        // run the matrix maddiplication calculation
        madd<<<grid_size, block_size>>>( A_device, B_device, m_dim, 0 /*no offset*/);
        cudaDeviceSynchronize();

        // retrieve results
        cudaMemcpy(B_host, B_device, m_tot * sizeof(float), cudaMemcpyDeviceToHost);
    }

    // end the overall timer
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&ms, startEvent, stopEvent);
    printf(" Time total (ms): %f\n", ms);
    printf("\n");

    //~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*
    //~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*
    //~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*

    // printf(" --- Overlapping Run ---\n");

    // // we store in single dimension where row is assumed first
    // float* A_host_overlap = new float[n_repeat * m_tot];
    // float* B_host_overlap = new float[n_repeat * m_tot];

    // // we set n_repeat times
    // for (int i = 0; i < n_repeat; ++i)
    // {
    //     // for now for simplicity we set all to 1 for A
    //     for (int ii = 0; ii < m_tot; ++ii)
    //     {
    //         A_host_overlap[ii + i * m_tot] = 1;
    //     }
    // }

    // // create pointer to the device matrix input
    // float* A_overlap;
    // float* B_overlap;

    // // allocate memory to the pointer
    // cudaMalloc((void**) &A_overlap, 4 * m_tot * sizeof(float));
    // cudaMalloc((void**) &B_overlap, 4 * m_tot * sizeof(float));

    // // start the overall timer
    // cudaEventRecord(startEvent, 0);

    // for (int i = 0; i < n_repeat; ++i)
    // {
    //     int offset = i * m_tot;
    //     // copy the host values to device memory
    //     cudaMemcpyAsync(&A_overlap[offset], &A_host_overlap[offset], m_tot * sizeof(float), cudaMemcpyHostToDevice, stream[i]);

    //     // run the matrix maddiplication calculation
    //     madd<<<grid_size, block_size, 0, stream[i]>>>( A_device, B_device, m_dim, i /*no offset*/);

    //     // retrieve results
    //     cudaMemcpyAsync(&B_host_overlap[offset], &B_overlap[offset], m_tot * sizeof(float), cudaMemcpyDeviceToHost, stream[i]);
    // }

    // // end the overall timer
    // cudaEventRecord(stopEvent, 0);
    // cudaEventSynchronize(stopEvent);
    // cudaEventElapsedTime(&ms, startEvent, stopEvent);
    // printf(" Time total (ms): %f\n", ms);
    // printf("\n");


    // cleanup
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
    cudaFree(A_device);
    cudaFree(B_device);
    free(A_host);
    free(B_host);

    // for (int i = 0; i < n_repeat; ++i)
    //     cudaStreamDestroy(stream[i]);
    // cudaFree(A_overlap);
    // cudaFree(B_overlap);
    // free(A_host_overlap);
    // free(B_host_overlap);

}
