// Alberto Taddei & Thies Weel
// Heston Model Monte Carlo - Euler paths for visualization
// - Simula come heston_1.cu
// - In più salva i primi N_SAVE path completi S_t in paths.csv

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <curand_kernel.h>

#define THREADS_PER_BLOCK 256
#define NUM_BLOCKS 1024
#define TOTAL_PATHS (THREADS_PER_BLOCK * NUM_BLOCKS)  // 262,144 paths

// Quanti path completi salvare per la visualizzazione
#define N_SAVE 50

// Model parameters
#define S0 1.0f
#define v0 0.1f
#define r  0.0f
#define kappa 0.5f
#define theta 0.1f
#define sigma 0.3f
#define rho 0.0f   // per ora 0 come nello step 1
#define T 1.0f
#define K 1.0f
#define M 1000     // Δt = 1/1000

//-----------------------------
// Error handling
//-----------------------------
void testCUDA(cudaError_t error, const char *file, int line) {
    if (error != cudaSuccess) {
        printf("CUDA error at %s:%d: %s\n",
               file, line, cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}
#define testCUDA(error) (testCUDA(error, __FILE__, __LINE__))

//-----------------------------
// Euler kernel
// payoffs: payoff terminale per pricing
// paths:   path completi S_t per i primi N_SAVE thread
//-----------------------------
__global__ void heston_euler_kernel(
    float *payoffs,
    float *paths,
    unsigned long seed,
    int use_abs)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= TOTAL_PATHS) return;

    // RNG
    curandState state;
    curand_init(seed, idx, 0, &state);

    float dt      = T / (float)M;
    float sqrt_dt = sqrtf(dt);
    float sqrt_1_minus_rho2 = sqrtf(1.0f - rho * rho);

    float S = S0;
    float v = v0;

    // Se questo thread è tra i primi N_SAVE, salviamo S_0
    if (idx < N_SAVE) {
        paths[idx * (M + 1) + 0] = S0;
    }

    for (int step = 0; step < M; ++step) {
        float G1 = curand_normal(&state);
        float G2 = curand_normal(&state);

        float dZ = rho * G1 + sqrt_1_minus_rho2 * G2;

        // S_{t+Δt}
        S = S + r * S * dt
              + sqrtf(fmaxf(v, 0.0f)) * S * sqrt_dt * dZ;

        // v_{t+Δt}
        float v_new = v + kappa * (theta - v) * dt
                        + sigma * sqrtf(fmaxf(v, 0.0f)) * sqrt_dt * G1;

        if (use_abs)
            v = fabsf(v_new);         // g(x) = |x|
        else
            v = fmaxf(v_new, 0.0f);   // g(x) = (x)+

        // Salva S_{t+Δt} per i primi N_SAVE path
        if (idx < N_SAVE) {
            paths[idx * (M + 1) + (step + 1)] = S;
        }
    }

    payoffs[idx] = fmaxf(S - K, 0.0f);
}

//-----------------------------
// Reduction kernel per sommare i payoffs
//-----------------------------
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

//-----------------------------
// Host function: simulazione + salvataggio paths
//-----------------------------
float heston_euler_simulation_with_paths(int use_abs)
{
    float *d_payoffs, *d_partial_sums, *d_paths;

    testCUDA(cudaMalloc(&d_payoffs,      TOTAL_PATHS * sizeof(float)));
    testCUDA(cudaMalloc(&d_partial_sums, NUM_BLOCKS   * sizeof(float)));
    testCUDA(cudaMalloc(&d_paths,        N_SAVE * (M + 1) * sizeof(float)));

    unsigned long seed = 12345UL;

    // Timer
    cudaEvent_t start, stop;
    testCUDA(cudaEventCreate(&start));
    testCUDA(cudaEventCreate(&stop));
    testCUDA(cudaEventRecord(start, 0));

    // Kernel
    heston_euler_kernel<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(
        d_payoffs, d_paths, seed, use_abs);
    testCUDA(cudaGetLastError());

    // Reduction
    reduction_kernel<<<NUM_BLOCKS, THREADS_PER_BLOCK,
                       THREADS_PER_BLOCK * sizeof(float)>>>(
        d_payoffs, d_partial_sums, TOTAL_PATHS);
    testCUDA(cudaGetLastError());

    testCUDA(cudaEventRecord(stop, 0));
    testCUDA(cudaEventSynchronize(stop));

    float elapsed_ms;
    testCUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));

    // Somma finale su host
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

    // Copia path completi sul host
    float *h_paths = (float*)malloc(N_SAVE * (M + 1) * sizeof(float));
    testCUDA(cudaMemcpy(h_paths, d_paths,
                        N_SAVE * (M + 1) * sizeof(float),
                        cudaMemcpyDeviceToHost));

    // Salva in CSV: time,path0,path1,...,path{N_SAVE-1}
    FILE *csv = fopen("paths.csv", "w");
    if (!csv) {
        fprintf(stderr, "Errore apertura paths.csv\n");
    } else {
        float dt = T / (float)M;
        // Header
        fprintf(csv, "t");
        for (int j = 0; j < N_SAVE; ++j)
            fprintf(csv, ",path_%d", j);
        fprintf(csv, "\n");

        // Righe: t_k, S^{(0)}_k, ..., S^{(N_SAVE-1)}_k
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

    // Cleanup
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
// main
//-----------------------------
int main(void)
{
    printf("=============================================================\n");
    printf("Heston Model - Euler paths for visualization\n");
    printf("=============================================================\n");
    printf("Saving first %d paths (M=%d steps) to paths.csv\n", N_SAVE, M);

    // Usa g(x) = (x)+ per coerenza con il progetto
    heston_euler_simulation_with_paths(0);

    return 0;
}
