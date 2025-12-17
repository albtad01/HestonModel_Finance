// Alberto Taddei & Thies Weel
// Heston Model Monte Carlo - Simulates Euler paths for visualization

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <curand_kernel.h>

#define THREADS_PER_BLOCK 256
#define NUM_BLOCKS 1024
#define TOTAL_PATHS (THREADS_PER_BLOCK * NUM_BLOCKS)  // 262,144 paths

#define N_SAVE 50

#define S0 1.0f
#define v0 0.1f
#define r  0.0f
#define kappa 0.5f
#define theta 0.1f
#define sigma 0.3f
#define rho 0.0f   
#define T 1.0f
#define K 1.0f
#define M 1000     // delta_t = 1/1000

void testCUDA(cudaError_t error, const char *file, int line) {
    if (error != cudaSuccess) {
        printf("CUDA error at %s:%d: %s\n",
               file, line, cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}
#define testCUDA(error) (testCUDA(error, __FILE__, __LINE__))

__global__ void heston_euler_kernel(
    float *payoffs,
    float *paths,
    unsigned long seed,
    int use_abs)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= TOTAL_PATHS) return;

    curandState state;
    curand_init(seed, idx, 0, &state);

    float dt      = T / (float)M;
    float sqrt_dt = sqrtf(dt);
    float sqrt_1_minus_rho2 = sqrtf(1.0f - rho * rho);

    float S = S0;
    float v = v0;

    if (idx < N_SAVE) {
        paths[idx * (M + 1) + 0] = S0;
    }

    for (int step = 0; step < M; ++step) {
        float G1 = curand_normal(&state);
        float G2 = curand_normal(&state);

        float dZ = rho * G1 + sqrt_1_minus_rho2 * G2;

        S = S + r * S * dt
              + sqrtf(fmaxf(v, 0.0f)) * S * sqrt_dt * dZ;

        float v_new = v + kappa * (theta - v) * dt
                        + sigma * sqrtf(fmaxf(v, 0.0f)) * sqrt_dt * G1;

        if (use_abs)
            v = fabsf(v_new);         
        else
            v = fmaxf(v_new, 0.0f);   

        if (idx < N_SAVE) {
            paths[idx * (M + 1) + (step + 1)] = S;
        }
    }

    payoffs[idx] = fmaxf(S - K, 0.0f);
}

__global__ void reduction_kernel(float *payoffs,
                                 float *partial_sums,
                                 int N)
{
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (idx < N) ? payoffs[idx] : 0.0f;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial_sums[blockIdx.x] = sdata[tid];
    }
}

float heston_euler_simulation_with_paths(int use_abs)
{
    float *d_payoffs, *d_partial_sums, *d_paths;

    testCUDA(cudaMalloc(&d_payoffs,      TOTAL_PATHS * sizeof(float)));
    testCUDA(cudaMalloc(&d_partial_sums, NUM_BLOCKS   * sizeof(float)));
    testCUDA(cudaMalloc(&d_paths,        N_SAVE * (M + 1) * sizeof(float)));

    unsigned long seed = 12345UL;

    cudaEvent_t start, stop;
    testCUDA(cudaEventCreate(&start));
    testCUDA(cudaEventCreate(&stop));
    testCUDA(cudaEventRecord(start, 0));

    heston_euler_kernel<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(
        d_payoffs, d_paths, seed, use_abs);
    testCUDA(cudaGetLastError());

    reduction_kernel<<<NUM_BLOCKS, THREADS_PER_BLOCK,
                       THREADS_PER_BLOCK * sizeof(float)>>>(
        d_payoffs, d_partial_sums, TOTAL_PATHS);
    testCUDA(cudaGetLastError());

    testCUDA(cudaEventRecord(stop, 0));
    testCUDA(cudaEventSynchronize(stop));

    float elapsed_ms;
    testCUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));

    float *h_partial_sums = (float*)malloc(NUM_BLOCKS * sizeof(float));
    testCUDA(cudaMemcpy(h_partial_sums, d_partial_sums,
                        NUM_BLOCKS * sizeof(float),
                        cudaMemcpyDeviceToHost));

    float total_sum = 0.0f;
    for (int i = 0; i < NUM_BLOCKS; ++i)
        total_sum += h_partial_sums[i];

    float option_price = total_sum / TOTAL_PATHS;

    printf("\n=== Euler (g = %s) ===\n", use_abs ? "|x|" : "(x)+");
    printf("Estimated price E[(S_1 - 1)+] = %.6f\n", option_price);
    printf("Execution time: %.3f ms\n", elapsed_ms);

    float *h_paths = (float*)malloc(N_SAVE * (M + 1) * sizeof(float));
    testCUDA(cudaMemcpy(h_paths, d_paths,
                        N_SAVE * (M + 1) * sizeof(float),
                        cudaMemcpyDeviceToHost));

    FILE *csv = fopen("results/paths.csv", "w");
    if (!csv) {
        fprintf(stderr, "Errore apertura paths.csv\n");
    } else {
        float dt = T / (float)M;
        fprintf(csv, "t");
        for (int j = 0; j < N_SAVE; ++j)
            fprintf(csv, ",path_%d", j);
        fprintf(csv, "\n");

        for (int k = 0; k <= M; ++k) {
            float t = k * dt;
            fprintf(csv, "%.6f", t);
            for (int j = 0; j < N_SAVE; ++j) {
                float S_jk = h_paths[j * (M + 1) + k];
                fprintf(csv, ",%.6f", S_jk);
            }
            fprintf(csv, "\n");
        }
        fclose(csv);
        printf("Saved %d paths to paths.csv (N_SAVE=%d, M=%d)\n",
               N_SAVE, N_SAVE, M);
    }

    free(h_partial_sums);
    free(h_paths);
    testCUDA(cudaFree(d_payoffs));
    testCUDA(cudaFree(d_partial_sums));
    testCUDA(cudaFree(d_paths));
    testCUDA(cudaEventDestroy(start));
    testCUDA(cudaEventDestroy(stop));

    return option_price;
}

//-----------------------------
int main(void)
{
    printf("=============================================================\n");
    printf("Heston Model - Euler paths for visualization\n");
    printf("=============================================================\n");
    printf("Saving first %d paths (M=%d steps) to paths.csv\n", N_SAVE, M);

    heston_euler_simulation_with_paths(0);

    return 0;
}
