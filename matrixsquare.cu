#include <cstdlib>
#include <iostream>
#include <random>
#include <chrono>
using namespace std::chrono;

__global__ void mult(float* A,
                     float* B,
                     int m_dim,
                     int ioffset)
{
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    int col = blockDim.y * blockIdx.y + threadIdx.y;

    // If out of bounds do nothing and return
    if (row >= m_dim || col >= m_dim)
        return;

    float Bval = 0;
    for (int ii = 0; ii < m_dim; ++ii)
    {
        float Aval_row = A[m_dim * row + ii + ioffset * m_dim * m_dim];
        float Aval_col = A[m_dim * ii + row + ioffset * m_dim * m_dim];
        Bval += Aval * Aval;
    }
    B[row * m_dim + col + ioffset * m_dim * m_dim] = Bval;
}

int main(int argc, char** argv)
{

    std::cout << "################################" << std::endl;
    std::cout << "#                              #" << std::endl;
    std::cout << "#       Matrix Square          #" << std::endl;
    std::cout << "#     (Overlap Transfer)       #" << std::endl;
    std::cout << "#                              #" << std::endl;
    std::cout << "################################" << std::endl;

    // goal: Perform N times A * A = B matrix squaring
    const int n_repeat = 4;

    // A matrix dimension and total element definition
    const int m_dim = 1024;
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

    // variable to read out time
    float ms;

    // we store in single dimension where row is assumed first
    float* A_host = new float[m_tot];
    float* B_host = new float[m_tot];

    // for now for simplicity we set all to 1 for A
    for (int ii = 0; ii < m_dim; ++ii)
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
    mult<<<grid_size, block_size>>>( A_device, B_device, m_dim, 0 /*no offset*/);

    // start the overall timer
    cudaEventRecord(startEvent, 0);

    for (int i = 0 ; i < n_repeat; ++i)
    {
        // copy the host values to device memory
        cudaMemcpy(A_device, A_host, m_tot * sizeof(float), cudaMemcpyHostToDevice);

        // run the matrix multiplication calculation
        mult<<<grid_size, block_size>>>( A_device, B_device, m_dim, 0 /*no offset*/);
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

    ////~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*
    ////~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*
    ////~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*

    //printf(" --- Overlapping Run ---\n");

    //// we store in single dimension where row is assumed first
    //float* A_host_overlap = new float[n_repeat * A_ntot];
    //float* B_host_overlap = new float[n_repeat * B_ntot];
    //float* C_host_overlap = new float[n_repeat * C_ntot];

    //// we set n_repeat times
    //for (int i = 0; i < n_repeat; ++i)
    //{
    //    // for now for simplicity we set all to 1 for A
    //    for (int ii = 0; ii < A_ntot; ++ii)
    //    {
    //        A_host_overlap[ii + i * A_ntot] = 1;
    //    }

    //    // and all value to 2 for B
    //    for (int ii = 0; ii < B_ntot; ++ii)
    //    {
    //        B_host_overlap[ii + i * B_ntot] = 2;
    //    }
    //}

    //// create pointer to the device matrix input
    //float* A_overlap;
    //float* B_overlap;
    //float* C_overlap;

    //// allocate memory to the pointer
    //cudaMalloc((void**) &A_overlap, 4 * A_ntot * sizeof(float));
    //cudaMalloc((void**) &B_overlap, 4 * B_ntot * sizeof(float));
    //cudaMalloc((void**) &C_overlap, 4 * C_ntot * sizeof(float));

    //// create cuda streams
    //cudaStream_t stream[n_repeat];
    //for (int i = 0; i < n_repeat; ++i)
    //{
    //    cudaStreamCreate(&stream[i]);
    //}

    //// start the overall timer
    //cudaEventRecord(startEvent, 0);

    //auto start = high_resolution_clock::now();

    //for (int i = 0; i < n_repeat; ++i)
    //{
    //    // copy the host values to device memory
    //    cudaMemcpyAsync(&A_overlap[i * A_ntot], &A_host_overlap[i * A_ntot], A_ntot * sizeof(float), cudaMemcpyHostToDevice, stream[i]);
    //    cudaMemcpyAsync(&B_overlap[i * B_ntot], &B_host_overlap[i * B_ntot], B_ntot * sizeof(float), cudaMemcpyHostToDevice, stream[i]);

    //    // run the matrix multiplication calculation
    //    mult<<<grid_size, block_size, 0, stream[i]>>>(
    //        A_overlap, B_overlap, C_overlap,
    //        A_nrow, A_ncol,
    //        B_nrow, B_ncol,
    //        C_nrow, C_ncol, i);

    //    // retrieve results
    //    cudaMemcpyAsync(&C_host_overlap[i * C_ntot], &C_overlap[i * C_ntot], C_ntot * sizeof(float), cudaMemcpyDeviceToHost, stream[i]);
    //}

    //// end the overall timer
    //cudaEventRecord(stopEvent, 0);
    //cudaEventSynchronize(stopEvent);
    //cudaEventElapsedTime(&ms, startEvent, stopEvent);
    //printf(" Time total (ms): %f\n", ms);
    //printf("\n");


    // cleanup
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
    // for (int i = 0; i < n_repeat; ++i)
    //     cudaStreamDestroy(stream[i]);
    cudaFree(A_device);
    cudaFree(B_device);
    // cudaFree(A_overlap);
    // cudaFree(B_overlap);
    free(A_host);
    free(B_host);
    // free(A_host_overlap);
    // free(B_host_overlap);

}
