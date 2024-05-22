#include <iostream>
#include <map>
#include <vector>
#include "csv.h"
using namespace csv;

#define checkCuda(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}


const float k2Rinv1GeVf = (2.99792458e-3 * 3.8) / 2;
const float sinAlphaMax = 0.95;
const float ptCut = 1.5;

__global__ void create_minidoublets(float* xs, float* ys, float* zs,
                                    float* md_xs_l, float* md_ys_l, float* md_zs_l,
                                    float* md_xs_u, float* md_ys_u, float* md_zs_u,
                                    int* n_mds_d,
                                    const int n_lower_det_id,
                                    const int n_max_hits,
                                    const int n_max_mds)
{
    int lower_det_id = blockDim.x * blockIdx.x + threadIdx.x;
    if (lower_det_id < n_lower_det_id)
    {
        for (int ii = 0; ii < n_max_hits; ++ii)
        {
            float xl = xs[(2 * lower_det_id) * n_max_hits + ii];
            float yl = ys[(2 * lower_det_id) * n_max_hits + ii];
            float zl = zs[(2 * lower_det_id) * n_max_hits + ii];
            float rl = sqrt(xl * xl + yl * yl);
            for (int jj = 0; jj < n_max_hits; ++jj)
            {
                float xu = xs[(2 * lower_det_id + 1) * n_max_hits + jj];
                float yu = ys[(2 * lower_det_id + 1) * n_max_hits + jj];
                float zu = zs[(2 * lower_det_id + 1) * n_max_hits + jj];

                float xd = xu - xl;
                float yd = yu - yl;

                // using dot product we compute the phi in azimuthal angle
                float dphi = acosf((xl * xd + yl * yd) / sqrt(xl * xl + yl * yl) / sqrt(xd * xd + yd * yd));
                const float dphi_cut = asinf(min(rl * k2Rinv1GeVf / ptCut, sinAlphaMax));

                // using geometry we find the threshold for the cut value
                if (abs(dphi) < dphi_cut)
                {
                    int imd = atomicAdd(&n_mds_d[lower_det_id], 1);
                    // this is needed to not bleed into next section of memory
                    if (imd < 100)
                    {
                        md_xs_l[lower_det_id * n_max_mds + imd] = xl;
                        md_ys_l[lower_det_id * n_max_mds + imd] = yl;
                        md_zs_l[lower_det_id * n_max_mds + imd] = zl;
                        md_xs_u[lower_det_id * n_max_mds + imd] = xu;
                        md_ys_u[lower_det_id * n_max_mds + imd] = yu;
                        md_zs_u[lower_det_id * n_max_mds + imd] = zu;
                    }
                }
            }
        }
    }
    return;
}

int main()
{

    // Problem statement:
    //
    // We have a list of hits in an event stored in "hits.csv".
    //
    // Each hit has x, y, z position and a "det_id"
    //
    // "det_id" stands for "detector ID" and is a unique
    // identifying integer for each detector module
    //
    // Each detector module has a paired detector module
    // where they are very closely placed together
    //
    // We want to correlate two hits, one from each paired module,
    // and compute the angle between two vectors, (origin, hit1)
    // and (hit1, hit2) (in xy plane)
    //
    // This angle will be large if the two hits came from
    // a low momentum charged particle
    //
    // This angle will be small if the two hits came from
    // a high momentum charged particle
    //
    // In general, in our experiments, we care more about high
    // momentum charged particle, therefore we can place a requirement
    // that a "good pair of hits" have angle smaller than certain
    // threshold
    //
    // We will use GPU to go through the combinatorics and
    // compute these angles for given pair of hits and
    // make the requirement that it is less than some threshold
    //



    // std::cout << "######################" << std::endl;
    // std::cout << "#                    #" << std::endl;
    // std::cout << "#    Mini-Doublet    #" << std::endl;
    // std::cout << "#                    #" << std::endl;
    // std::cout << "######################" << std::endl;

    // Reading inputs
    CSVReader reader("hits.csv");

    // Parsing total number of hits data
    int n_data = reader.n_rows();

    // We will count how many hits are in per det_id
    // So we can compute an n_max_hits to allocate
    std::map<int, int> n_hits_per_det_id;
    for (CSVRow& row: reader)
    {
        n_hits_per_det_id[row["det_id"].get<int>()] += 1;
    }

    // Find the max value
    int n_max_hits = 0;
    for (const auto& pair : n_hits_per_det_id)
    {
        if (pair.second > n_max_hits)
        {
            n_max_hits = pair.second;
        }
    }

    // Find how many unique det_ids we have
    int n_det_id = n_hits_per_det_id.size();
    int n_lower_det_id = n_det_id / 2.;

    // We will allocate 10 * n_max_hits * n_det_id memory
    // Why 10 * ? because some other events may have more hits
    int n_total_hits = 10 * n_det_id * n_max_hits;

    // std::cout <<  " n_det_id: " << n_det_id <<  std::endl;
    // std::cout <<  " n_max_hits: " << n_max_hits <<  std::endl;

    // We will also keep track of unique *lower* detid -> idx
    std::map<int, int> map_lower_detid_idx;
    int idx = 0;
    for (const auto& pair : n_hits_per_det_id)
    {
        int det_id = pair.first;
        if (det_id % 2 == 0)
        {
            map_lower_detid_idx[pair.first] = idx;
            idx++;
        }
    }

    // We create a pointer to hold the x's
    float* xs;
    float* ys;
    float* zs;

    // Allocate host memory where we will load the inputs
    checkCuda(cudaMallocHost((void**) &xs, n_total_hits * sizeof(float)));
    checkCuda(cudaMallocHost((void**) &ys, n_total_hits * sizeof(float)));
    checkCuda(cudaMallocHost((void**) &zs, n_total_hits * sizeof(float)));

    // Load the inputs
    std::map<int, int> map_detid_counter;
    reader = CSVReader("hits.csv");
    for (CSVRow& row: reader)
    {
        int det_id = row["det_id"].get<int>();
        int isupper = det_id % 2;
        int lower_det_id = isupper ? det_id - 1 : det_id;
        int lower_det_id_idx = map_lower_detid_idx[lower_det_id];
        int det_id_counter = map_detid_counter[det_id];
        float x = row["x"].get<float>();
        float y = row["y"].get<float>();
        float z = row["z"].get<float>();
        int hit_idx = (2 * lower_det_id_idx + isupper) * n_max_hits + det_id_counter;
        xs[hit_idx] = x;
        ys[hit_idx] = y;
        zs[hit_idx] = z;
        map_detid_counter[det_id]++;
    }

    // Allocate device memory where we will load the inputs
    // Declaring pointers to host memory
    float* xs_d;
    float* ys_d;
    float* zs_d;

    // Allocate host memory where we will load the inputs
    checkCuda(cudaMalloc((void**) &xs_d, n_total_hits * sizeof(float)));
    checkCuda(cudaMalloc((void**) &ys_d, n_total_hits * sizeof(float)));
    checkCuda(cudaMalloc((void**) &zs_d, n_total_hits * sizeof(float)));

    // Copy the input to device
    checkCuda(cudaMemcpy(xs_d, xs, n_total_hits * sizeof(float), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(ys_d, ys, n_total_hits * sizeof(float), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(zs_d, zs, n_total_hits * sizeof(float), cudaMemcpyHostToDevice));

    // Once create mini-doublets (md) we will save them
    float* md_xs_l_d;
    float* md_ys_l_d;
    float* md_zs_l_d;
    float* md_xs_u_d;
    float* md_ys_u_d;
    float* md_zs_u_d;
    int* n_mds_d;

    // We will save up to n_max_md
    int n_max_mds = 100;
    int n_total_mds = n_max_mds * n_lower_det_id;

    // Allocate host memory where we will load the inputs
    checkCuda(cudaMalloc((void**) &md_xs_l_d, n_total_mds * sizeof(float)));
    checkCuda(cudaMalloc((void**) &md_ys_l_d, n_total_mds * sizeof(float)));
    checkCuda(cudaMalloc((void**) &md_zs_l_d, n_total_mds * sizeof(float)));
    checkCuda(cudaMalloc((void**) &md_xs_u_d, n_total_mds * sizeof(float)));
    checkCuda(cudaMalloc((void**) &md_ys_u_d, n_total_mds * sizeof(float)));
    checkCuda(cudaMalloc((void**) &md_zs_u_d, n_total_mds * sizeof(float)));
    checkCuda(cudaMalloc((void**) &n_mds_d, n_lower_det_id * sizeof(int)));

    int block_size = 256;
    int grid_size = int(n_lower_det_id - 0.5) / block_size + 1;
    create_minidoublets<<<grid_size, block_size>>>(xs_d, ys_d, zs_d,
                              md_xs_l_d, md_ys_l_d, md_zs_l_d,
                              md_xs_u_d, md_ys_u_d, md_zs_u_d,
                              n_mds_d,
                              n_lower_det_id,
                              n_max_hits,
                              n_max_mds);

    cudaDeviceSynchronize();

    float* md_xs_l;
    float* md_ys_l;
    float* md_zs_l;
    float* md_xs_u;
    float* md_ys_u;
    float* md_zs_u;
    int* n_mds;

    // Allocate host memory where we will read out the outputs
    checkCuda(cudaMallocHost((void**) &md_xs_l, n_total_mds * sizeof(float)));
    checkCuda(cudaMallocHost((void**) &md_ys_l, n_total_mds * sizeof(float)));
    checkCuda(cudaMallocHost((void**) &md_zs_l, n_total_mds * sizeof(float)));
    checkCuda(cudaMallocHost((void**) &md_xs_u, n_total_mds * sizeof(float)));
    checkCuda(cudaMallocHost((void**) &md_ys_u, n_total_mds * sizeof(float)));
    checkCuda(cudaMallocHost((void**) &md_zs_u, n_total_mds * sizeof(float)));
    checkCuda(cudaMallocHost((void**) &n_mds, n_lower_det_id * sizeof(int)));

    // copy back to read
    checkCuda(cudaMemcpy(md_xs_l, md_xs_l_d, n_total_mds * sizeof(float), cudaMemcpyDeviceToHost));
    checkCuda(cudaMemcpy(md_ys_l, md_ys_l_d, n_total_mds * sizeof(float), cudaMemcpyDeviceToHost));
    checkCuda(cudaMemcpy(md_zs_l, md_zs_l_d, n_total_mds * sizeof(float), cudaMemcpyDeviceToHost));
    checkCuda(cudaMemcpy(md_xs_u, md_xs_u_d, n_total_mds * sizeof(float), cudaMemcpyDeviceToHost));
    checkCuda(cudaMemcpy(md_ys_u, md_ys_u_d, n_total_mds * sizeof(float), cudaMemcpyDeviceToHost));
    checkCuda(cudaMemcpy(md_zs_u, md_zs_u_d, n_total_mds * sizeof(float), cudaMemcpyDeviceToHost));
    checkCuda(cudaMemcpy(n_mds, n_mds_d, n_lower_det_id * sizeof(int), cudaMemcpyDeviceToHost));

    std::cout << "x1,y1,z1,x2,y2,z2" << std::endl;
    for (int idet = 0; idet < n_lower_det_id; ++idet)
    {
        for (int imd = 0; imd < n_mds[idet]; ++imd)
        {
            std::cout << md_xs_l[idet * n_max_mds + imd] << ",";
            std::cout << md_ys_l[idet * n_max_mds + imd] << ",";
            std::cout << md_zs_l[idet * n_max_mds + imd] << ",";
            std::cout << md_xs_u[idet * n_max_mds + imd] << ",";
            std::cout << md_ys_u[idet * n_max_mds + imd] << ",";
            std::cout << md_zs_u[idet * n_max_mds + imd] << std::endl;
        }
    }

    return 0;
}
