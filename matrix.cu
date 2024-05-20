#include <cstdlib>
#include <iostream>
#include <random>
#include <chrono>
using namespace std::chrono;

#define myInt_t unsigned long long

__global__ void mult(double* A,
                     double* B,
                     double* C,
                     myInt_t A_nrow,
                     myInt_t A_ncol,
                     myInt_t B_nrow,
                     myInt_t B_ncol,
                     myInt_t C_nrow,
                     myInt_t C_ncol,
                     int ioffset)
{
    myInt_t row = blockDim.x * blockIdx.x + threadIdx.x;
    myInt_t col = blockDim.y * blockIdx.y + threadIdx.y;

    // If out of bounds do nothing and return
    if (row >= A_nrow || col >= B_ncol)
        return;

    for (myInt_t ii = 0; ii < A_nrow; ++ii)
    {
        C[row * C_ncol + col + ioffset * C_nrow * C_ncol] +=
            A[A_ncol * row + ii + ioffset * A_nrow * A_ncol] *
            B[B_ncol * ii + col + ioffset * B_nrow * B_ncol];
    }
}

int main(int argc, char** argv)
{

    std::cout << "################################" << std::endl;
    std::cout << "#                              #" << std::endl;
    std::cout << "#    Matrix Multiplication     #" << std::endl;
    std::cout << "#     (Overlap Transfer)       #" << std::endl;
    std::cout << "#                              #" << std::endl;
    std::cout << "################################" << std::endl;

    // goal: Perform N times A * B = C matrix multiplication
    myInt_t n_repeat = 4;

    // A matrix dimension and total element definition
    myInt_t A_nrow = 5;
    myInt_t A_ncol = 200;
    myInt_t A_ntot = A_nrow * A_ncol;

    // B matrix dimension and total element definition
    myInt_t B_nrow = 200;
    myInt_t B_ncol = 5;
    myInt_t B_ntot = B_nrow * B_ncol;

    // C matrix dimension and total element definition
    myInt_t C_nrow = A_nrow;
    myInt_t C_ncol = B_ncol;
    myInt_t C_ntot = C_nrow * C_ncol;

    // we will perform each element as one thread
    // 4 element x 4 element will be computed per thread block
    myInt_t n_thread_dim = 16;

    // then the block dimensions are defined
    dim3 blockDim(n_thread_dim, n_thread_dim, 1);

    // compute number of blocks in each dimension
    myInt_t grid_size_x = int(C_nrow - 0.5) / n_thread_dim + 1;
    myInt_t grid_size_y = int(C_ncol - 0.5) / n_thread_dim + 1;

    // then for grid size needs to be computed to cover the entire elements
    dim3 gridDim(grid_size_x, grid_size_y, 1);

    // define cuda events
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    cudaEvent_t startTask, stopTask;
    cudaEventCreate(&startTask);
    cudaEventCreate(&stopTask);

    // variable to read out time
    float ms;

    // we store in single dimension where row is assumed first
    double* A_host = new double[A_ntot];
    double* B_host = new double[B_ntot];
    double* C_host = new double[C_ntot];

    // for now for simplicity we set all to 1 for A
    for (myInt_t ii = 0; ii < A_ntot; ++ii)
    {
        A_host[ii] = 1;
    }

    // and all value to 2 for B
    for (myInt_t ii = 0; ii < B_ntot; ++ii)
    {
        B_host[ii] = 2;
    }

    //~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*
    //~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*
    //~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*

    printf(" --- Sequential Run ---\n");

    // create pointer to the device matrix input
    double* A_device;
    double* B_device;
    double* C_device;

    // allocate memory to the pointer
    cudaMalloc((void**) &A_device, A_ntot * sizeof(double));
    cudaMalloc((void**) &B_device, B_ntot * sizeof(double));
    cudaMalloc((void**) &C_device, C_ntot * sizeof(double));

    // warm up run
    mult<<<gridDim, blockDim>>>(
        A_device, B_device, C_device,
        A_nrow, A_ncol,
        B_nrow, B_ncol,
        C_nrow, C_ncol, 0 /*no offset*/);

    // start the overall timer
    cudaEventRecord(startEvent, 0);

    for (int i = 0 ; i < n_repeat; ++i)
    {
        // copy the host values to device memory
        // cudaEventRecord(startTask, 0);
        cudaMemcpy(A_device, A_host, A_ntot * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(B_device, B_host, B_ntot * sizeof(double), cudaMemcpyHostToDevice);
        // cudaEventRecord(stopTask, 0);
        // cudaEventSynchronize(stopTask);
        // cudaEventElapsedTime(&ms, startTask, stopTask);
        // printf(" Time tx    (ms): %f\n", ms);

        // run the matrix multiplication calculation
        // cudaEventRecord(startTask, 0);
        mult<<<gridDim, blockDim>>>(
            A_device, B_device, C_device,
            A_nrow, A_ncol,
            B_nrow, B_ncol,
            C_nrow, C_ncol, 0 /*no offset*/);
        cudaDeviceSynchronize();
        // cudaEventRecord(stopTask, 0);
        // cudaEventSynchronize(stopTask);
        // cudaEventElapsedTime(&ms, startTask, stopTask);
        // printf(" Time exec  (ms): %f\n", ms);

        // retrieve results
        // cudaEventRecord(startTask, 0);
        cudaMemcpy(C_host, C_device, C_ntot * sizeof(double), cudaMemcpyDeviceToHost);
        // cudaEventRecord(stopTask, 0);
        // cudaEventSynchronize(stopTask);
        // cudaEventElapsedTime(&ms, startTask, stopTask);
        // printf(" Time rx    (ms): %f\n", ms);
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

    printf(" --- Overlapping Run ---\n");

    // we store in single dimension where row is assumed first
    double* A_host_overlap = new double[n_repeat * A_ntot];
    double* B_host_overlap = new double[n_repeat * B_ntot];
    double* C_host_overlap = new double[n_repeat * C_ntot];

    // we set n_repeat times
    for (int i = 0; i < n_repeat; ++i)
    {
        // for now for simplicity we set all to 1 for A
        for (myInt_t ii = 0; ii < A_ntot; ++ii)
        {
            A_host_overlap[ii + i * A_ntot] = 1;
        }

        // and all value to 2 for B
        for (myInt_t ii = 0; ii < B_ntot; ++ii)
        {
            B_host_overlap[ii + i * B_ntot] = 2;
        }
    }

    // create pointer to the device matrix input
    double* A_overlap;
    double* B_overlap;
    double* C_overlap;

    // allocate memory to the pointer
    cudaMalloc((void**) &A_overlap, 4 * A_ntot * sizeof(double));
    cudaMalloc((void**) &B_overlap, 4 * B_ntot * sizeof(double));
    cudaMalloc((void**) &C_overlap, 4 * C_ntot * sizeof(double));

    // create cuda streams
    cudaStream_t stream[n_repeat];
    for (int i = 0; i < n_repeat; ++i)
    {
        cudaStreamCreate(&stream[i]);
    }

    // start the overall timer
    cudaEventRecord(startEvent, 0);

    auto start = high_resolution_clock::now();

    for (int i = 0; i < n_repeat; ++i)
    {
        // copy the host values to device memory
        cudaMemcpyAsync(&A_overlap[i * A_ntot], &A_host_overlap[i * A_ntot], A_ntot * sizeof(double), cudaMemcpyHostToDevice, stream[i]);
        cudaMemcpyAsync(&B_overlap[i * B_ntot], &B_host_overlap[i * B_ntot], B_ntot * sizeof(double), cudaMemcpyHostToDevice, stream[i]);

        // run the matrix multiplication calculation
        mult<<<blockDim, gridDim, 0, stream[i]>>>(
            A_overlap, B_overlap, C_overlap,
            A_nrow, A_ncol,
            B_nrow, B_ncol,
            C_nrow, C_ncol, i);

        // retrieve results
        cudaMemcpyAsync(&C_host_overlap[i * C_ntot], &C_overlap[i * C_ntot], C_ntot * sizeof(double), cudaMemcpyDeviceToHost, stream[i]);
    }

    // end the overall timer
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&ms, startEvent, stopEvent);
    printf(" Time total (ms): %f\n", ms);
    printf("\n");

    // checking the output that it computed correctly
    printf(" --- Sanity Check ---\n");
    for (int i = 0; i < n_repeat; ++i)
    {
        for (myInt_t ii = 0; ii < C_nrow; ++ii)
        {
            for (myInt_t jj = 0; jj < C_ncol; ++jj)
            {
                double elem = C_host_overlap[ii * C_ncol + jj + C_ntot * i];
                printf("%9.2f ", elem);
            }
            printf("\n");
        }
        printf("\n");
    }


    // cleanup
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
    cudaEventDestroy(startTask);
    cudaEventDestroy(stopTask);
    for (int i = 0; i < n_repeat; ++i)
        cudaStreamDestroy(stream[i]);
    cudaFree(A_device);
    cudaFree(B_device);
    cudaFree(C_device);
    cudaFree(A_overlap);
    cudaFree(B_overlap);
    cudaFree(C_overlap);
    free(A_host);
    free(B_host);
    free(C_host);
    free(A_host_overlap);
    free(B_host_overlap);
    free(C_host_overlap);

}
