// Alberto Taddei & Thies Weel
// Heston Model Monte Carlo - Step 3: Performance Comparison
// Euler vs Almost Exact Scheme

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <curand_kernel.h>
#include <time.h>

#define THREADS_PER_BLOCK 256
#define NUM_BLOCKS        1024
#define TOTAL_PATHS       (THREADS_PER_BLOCK * NUM_BLOCKS)

// Fixed parameters
#define S0 1.0f
#define v0 0.1f
#define r  0.0f
#define T  1.0f
#define K  1.0f

// Ratio Bound: 2kappa theta  > sig2  for standard, or 20kappa theta  > sig2  as per Q3
// This means sig2 /(kappa theta ) < RATIO_BOUND
#define RATIO_BOUND 20.0f  // Change to 2.0f for standard Ratio Bound

// Function to catch CUDA errors
void testCUDA(cudaError_t error, const char *file, int line) {
    if (error != cudaSuccess) {
        printf("CUDA error at %s:%d: %s\n", file, line,
               cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}
#define testCUDA(error) (testCUDA(error, __FILE__, __LINE__))

// Structures
typedef struct {
    float kappa;
    float theta;
    float sigma;
} ParamSet;

typedef struct {
    float kappa;
    float theta;
    float sigma;
    float rho;
    int   M;
    const char* method_name;
    float time_ms;
    float price;
} BenchmarkResult;

// Random Number Generator init kernel
__global__ void init_rng_kernel(curandState *states, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < TOTAL_PATHS) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

// GAMMA DISTRIBUTION DEVICE FUNCTION from Paper [8]
__device__ float gamma_distribution(curandState *state, float alpha) {
    float boost_factor = 1.0f;

    // Case alpha  < 1
    if (alpha < 1.0f) {
        float u = curand_uniform(state);
        boost_factor = powf(u, 1.0f / alpha);
        alpha += 1.0f;
    }

    // Case alpha  >= 1
    float d = alpha - 1.0f / 3.0f;
    float c = 1.0f / sqrtf(9.0f * d);

    while (true) {
        float x, v;
        do {
            x = curand_normal(state);
            v = 1.0f + c * x;
        } while (v <= 0.0f);

        v = v * v * v;
        float u = curand_uniform(state);

        float x2 = x * x;
        if (u < 1.0f - 0.0331f * x2 * x2) {
            return d * v * boost_factor;
        }

        if (logf(u) < 0.5f * x2 + d * (1.0f - v + logf(v))) {
            return d * v * boost_factor;
        }
    }
}

// Kernel 1: EULER SCHEME
__global__ void heston_euler_kernel(
    float * __restrict__ payoffs,
    curandState * __restrict__ states,
    float kappa,
    float theta,
    float sigma,
    float rho,
    int   M)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= TOTAL_PATHS) return;

    curandState localState = states[idx];

    float dt          = T / (float)M;
    float sqrt_dt     = sqrtf(dt);
    float sqrt_1_rho2 = sqrtf(1.0f - rho * rho);

    float S = S0;
    float v = v0;

    for (int step = 0; step < M; ++step) {
        float G1 = curand_normal(&localState);
        float G2 = curand_normal(&localState);
        float dZ = rho * G1 + sqrt_1_rho2 * G2;

        S = S + r * S * dt
            + sqrtf(fmaxf(v, 0.0f)) * S * sqrt_dt * dZ;

        float v_new = v + kappa * (theta - v) * dt
                        + sigma * sqrtf(fmaxf(v, 0.0f)) * sqrt_dt * G1;
        v = fmaxf(v_new, 0.0f);
    }

    payoffs[idx] = fmaxf(S - K, 0.0f);
    states[idx] = localState;
}

// Kernel 2: ALMOST EXACT SCHEME
__global__ void heston_almost_exact_kernel(
    float * __restrict__ payoffs,
    curandState * __restrict__ states,
    float kappa,
    float theta,
    float sigma,
    float rho,
    int   M)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= TOTAL_PATHS) return;

    curandState localState = states[idx];

    float dt           = T / (float)M;
    float exp_kappa_dt = expf(-kappa * dt);

    float d     = 2.0f * kappa * theta / (sigma * sigma);
    float coeff = sigma * sigma * (1.0f - exp_kappa_dt) / (2.0f * kappa);

    float k0 = (-rho / sigma * kappa * theta) * dt;
    float k1 = (rho * kappa / sigma - 0.5f) * dt - rho / sigma;
    float k2 = rho / sigma;

    float log_S = logf(S0);
    float v     = v0;

    for (int step = 0; step < M; ++step) {
        float v_old = v;

        float lambda = 2.0f * kappa * exp_kappa_dt * v_old /
                       (sigma * sigma * (1.0f - exp_kappa_dt));
        unsigned int N     = curand_poisson(&localState, lambda);
        float alpha        = d + (float)N;      // alpha = d + N
        float gamma_sample = gamma_distribution(&localState, alpha);
        v = coeff * gamma_sample;

        // Independent Gaussian for the orthogonal Brownian part
        float G = curand_normal(&localState);

        // Diffusion term for almost-exact scheme
        float diffusion_term = sqrtf((1.0f - rho * rho) * fmaxf(v_old, 0.0f) * dt) * G;

        log_S = log_S + k0 + k1 * v_old + k2 * v + diffusion_term;
        }

    float S = expf(log_S);
    payoffs[idx] = fmaxf(S - K, 0.0f);

    states[idx] = localState;
}

// Reduction Kernel
__global__ void reduction_kernel(float * __restrict__ payoffs,
                                 float * __restrict__ partial_sums,
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

// Benchmark function
BenchmarkResult run_benchmark(
    float *d_payoffs,
    float *d_partial_sums,
    curandState *d_states,
    float *h_partial_sums,
    float kappa, float theta, float sigma, float rho, int M,
    bool use_almost_exact, const char* method_name)
{
    cudaEvent_t start, stop;
    testCUDA(cudaEventCreate(&start));
    testCUDA(cudaEventCreate(&stop));

    testCUDA(cudaEventRecord(start, 0));

    if (use_almost_exact) {
        heston_almost_exact_kernel<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(
            d_payoffs, d_states, kappa, theta, sigma, rho, M);
    } else {
        heston_euler_kernel<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(
            d_payoffs, d_states, kappa, theta, sigma, rho, M);
    }
    testCUDA(cudaGetLastError());

    reduction_kernel<<<NUM_BLOCKS, THREADS_PER_BLOCK,
                       THREADS_PER_BLOCK * sizeof(float)>>>(
        d_payoffs, d_partial_sums, TOTAL_PATHS);
    testCUDA(cudaGetLastError());

    testCUDA(cudaEventRecord(stop, 0));
    testCUDA(cudaEventSynchronize(stop));

    // Sum on host
    testCUDA(cudaMemcpy(h_partial_sums, d_partial_sums,
                        NUM_BLOCKS * sizeof(float),
                        cudaMemcpyDeviceToHost));

    double total = 0.0;
    for (int i = 0; i < NUM_BLOCKS; ++i)
        total += h_partial_sums[i];

    float price = (float)(total / TOTAL_PATHS);

    float time_ms;
    testCUDA(cudaEventElapsedTime(&time_ms, start, stop));

    testCUDA(cudaEventDestroy(start));
    testCUDA(cudaEventDestroy(stop));

    BenchmarkResult result;
    result.kappa       = kappa;
    result.theta       = theta;
    result.sigma       = sigma;
    result.rho         = rho;
    result.M           = M;
    result.method_name = method_name;
    result.time_ms     = time_ms;
    result.price       = price;

    return result;
}

// Parameter Generation
void generate_param_sets(ParamSet *params, int n_samples) {
    srand(time(NULL));

    int count    = 0;
    int attempts = 0;
    const int max_attempts = n_samples * 100;

    while (count < n_samples && attempts < max_attempts) {
        attempts++;

        float kappa = 0.1f + (float)rand() / RAND_MAX * (10.0f - 0.1f);
        float theta = 0.01f + (float)rand() / RAND_MAX * (0.5f - 0.01f);
        float sigma = 0.1f + (float)rand() / RAND_MAX * (1.0f - 0.1f);

        // Check Ratio Bound: RATIO_BOUND * kappa theta  > sig2 
        if (RATIO_BOUND * kappa * theta > sigma * sigma) {
            params[count].kappa = kappa;
            params[count].theta = theta;
            params[count].sigma = sigma;
            count++;
        }
    }

    if (count < n_samples) {
        printf("Warning: Only generated %d/%d valid parameter sets\n",
               count, n_samples);
    }
}

// Main Function
int main(void) {
    int M_values[] = {1000, 300, 100, 60, 30};
    int n_M = 5;

    printf("===============================================================\n");
    printf("Heston Model Monte Carlo - Step 3: Performance Comparison\n");
    printf("===============================================================\n");
    printf("Testing: kappa e [0.1, 10], theta e [0.01, 0.5], sigma e [0.1, 1]\n");
    printf("         rho e {-0.7, -0.3, 0, 0.3, 0.7}\n");
    printf("M values tested: {");
    for (int j = 0; j < n_M; ++j) {
        printf("%d%s", M_values[j], (j == n_M-1) ? "" : ", ");
    }
    printf("}  (dt = T/M)\n");
    printf("Constraint: %.0f*kappa*theta > sigma^2 (Ratio Bound: sigma^2/(kappa*theta) < %.0f)\n", 
           RATIO_BOUND, RATIO_BOUND);
    printf("Paths per test: %d\n", TOTAL_PATHS);
    printf("===============================================================\n\n");

    // Parameters
    const int N_PARAM_SETS = 30;
    ParamSet *param_sets =
        (ParamSet*)malloc(N_PARAM_SETS * sizeof(ParamSet));

    printf("Generating %d random parameter sets (kappa, theta, sigma)...\n",
           N_PARAM_SETS);
    generate_param_sets(param_sets, N_PARAM_SETS);

    printf("\nGenerated parameter sets:\n");
    printf("%-4s %-8s %-8s %-8s %-12s %-12s\n",
           "ID", "kappa", "theta", "sigma", "sigma^2/(kappa*theta)", "Margin");
    printf("---------------------------------------------------------------\n");
    for (int i = 0; i < N_PARAM_SETS; ++i) {
        float k = param_sets[i].kappa;
        float t = param_sets[i].theta;
        float s = param_sets[i].sigma;
        float ratio = (s*s)/(k*t);
        printf("%-4d %-8.3f %-8.4f %-8.3f %-12.3f %-12.3f\n",
               i+1, k, t, s, ratio, RATIO_BOUND - ratio);
    }
    printf("\n");

    float rho_values[] = {-0.7f, -0.3f, 0.0f, 0.3f, 0.7f};
    int n_rho = 5;

    int total_tests = N_PARAM_SETS * n_rho * n_M * 2;
    BenchmarkResult *results =
        (BenchmarkResult*)malloc(total_tests * sizeof(BenchmarkResult));

    // 2. Buffer allocation on device/host
    float *d_payoffs, *d_partial_sums;
    curandState *d_states_euler, *d_states_almost;
    testCUDA(cudaMalloc(&d_payoffs,      TOTAL_PATHS * sizeof(float)));
    testCUDA(cudaMalloc(&d_partial_sums, NUM_BLOCKS   * sizeof(float)));
    testCUDA(cudaMalloc(&d_states_euler, TOTAL_PATHS  * sizeof(curandState)));
    testCUDA(cudaMalloc(&d_states_almost,TOTAL_PATHS  * sizeof(curandState)));
    float *h_partial_sums = (float*)malloc(NUM_BLOCKS * sizeof(float));

    // 3. Random Number Generator initialization
    printf("Initializing RNG states...\n");
    init_rng_kernel<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_states_euler,  12345UL);
    init_rng_kernel<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_states_almost, 54321UL);
    testCUDA(cudaGetLastError());
    testCUDA(cudaDeviceSynchronize());

    // 4. Benchmark loop
    printf("Running %d benchmarks (this may take several minutes)...\n",
           total_tests);
    printf("Progress: [");
    fflush(stdout);

    int result_idx    = 0;
    int progress_step = total_tests / 50;
    if (progress_step == 0) progress_step = 1;

    for (int p = 0; p < N_PARAM_SETS; ++p) {
        float kappa = param_sets[p].kappa;
        float theta = param_sets[p].theta;
        float sigma = param_sets[p].sigma;

        float ratio = (sigma * sigma) / (kappa * theta);
        printf("\n[Param Set %d/%d] kappa=%.3f, theta=%.4f, sigma=%.3f -> sigma^2/(kappa*theta) = %.4f\n", 
               p+1, N_PARAM_SETS, kappa, theta, sigma, ratio);

        for (int rho_idx = 0; rho_idx < n_rho; ++rho_idx) {
            float rho = rho_values[rho_idx];

            for (int m_idx = 0; m_idx < n_M; ++m_idx) {
                int M = M_values[m_idx];

                // Euler
                results[result_idx++] = run_benchmark(
                    d_payoffs, d_partial_sums, d_states_euler, h_partial_sums,
                    kappa, theta, sigma, rho, M,
                    false, "Euler");

                // Almost Exact
                results[result_idx++] = run_benchmark(
                    d_payoffs, d_partial_sums, d_states_almost, h_partial_sums,
                    kappa, theta, sigma, rho, M,
                    true,  "Almost Exact");

                if (result_idx % progress_step == 0) {
                    printf("=");
                    fflush(stdout);
                }
            }
        }
    }
    printf("] Done!\n\n");

    // 5. ANALYSIS
    printf("===============================================================\n");
    printf("PERFORMANCE ANALYSIS\n");
    printf("===============================================================\n\n");

    // We have n_M different M values: M_values[0..n_M-1]
    float euler_time[64] = {0.0f};
    float ae_time[64]    = {0.0f};
    int   euler_cnt[64]  = {0};
    int   ae_cnt[64]     = {0};

    if (n_M > 64) {
        printf("Error: n_M too large for fixed arrays\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < total_tests; ++i) {
        BenchmarkResult res = results[i];

        // find index of res.M in M_values
        int midx = -1;
        for (int j = 0; j < n_M; ++j) {
            if (res.M == M_values[j]) { midx = j; break; }
        }
        if (midx < 0) continue; // should not happen

        if (strcmp(res.method_name, "Euler") == 0) {
            euler_time[midx] += res.time_ms;
            euler_cnt[midx]  += 1;
        } else { // "Almost Exact"
            ae_time[midx] += res.time_ms;
            ae_cnt[midx]  += 1;
        }
    }

    // Print table
    printf("%-8s %-10s %-18s %-18s %-10s\n",
        "M", "dt", "Euler avg (ms)", "AE avg (ms)", "AE/E");
    printf("--------------------------------------------------------------------------\n");

    for (int j = 0; j < n_M; ++j) {
        float dt = T / (float)M_values[j];

        float avgE  = (euler_cnt[j] > 0) ? (euler_time[j] / euler_cnt[j]) : 0.0f;
        float avgAE = (ae_cnt[j]    > 0) ? (ae_time[j]    / ae_cnt[j])    : 0.0f;
        float ratio = (avgE > 0.0f) ? (avgAE / avgE) : 0.0f;

        printf("%-8d %-10.6f %-18.3f %-18.3f %-10.3f\n",
            M_values[j], dt, avgE, avgAE, ratio);
    }
    printf("\n");

    // Impact: compare dt=1/30 (M=30) vs dt=1/1000 (M=1000) IF both exist
    int idx1000 = -1, idx30 = -1;
    for (int j = 0; j < n_M; ++j) {
        if (M_values[j] == 1000) idx1000 = j;
        if (M_values[j] == 30)   idx30   = j;
    }

    if (idx1000 >= 0 && idx30 >= 0) {
        float avgE1000  = euler_time[idx1000] / euler_cnt[idx1000];
        float avgE30    = euler_time[idx30]   / euler_cnt[idx30];
        float avgAE1000 = ae_time[idx1000]    / ae_cnt[idx1000];
        float avgAE30   = ae_time[idx30]      / ae_cnt[idx30];

        printf("Impact of using dt=1/30 (M=30) vs dt=1/1000 (M=1000):\n");
        printf("  Euler speedup:        %.2fx faster\n", avgE1000  / avgE30);
        printf("  Almost Exact speedup: %.2fx faster\n\n", avgAE1000 / avgAE30);
        printf("AE (M=30) vs Euler (M=1000) speedup: %.2fx\n", avgE1000 / avgAE30);
        printf("\n");
    } else {
        printf("Impact section skipped (need M=1000 and M=30 in M_values).\n\n");
    }

    // 6. CSV
    printf("Saving results to benchmark_results.csv...\n");
    FILE *csv = fopen("benchmark_results.csv", "w");
    fprintf(csv, "test_id,kappa,theta,sigma,rho,M,dt,method,time_ms,price\n");
    for (int i = 0; i < total_tests; ++i) {
        BenchmarkResult res = results[i];
        fprintf(csv,
                "%d,%.6f,%.6f,%.6f,%.6f,%d,%.6f,%s,%.3f,%.6f\n",
                i+1, res.kappa, res.theta, res.sigma, res.rho,
                res.M, T/(float)res.M,
                res.method_name, res.time_ms, res.price);
    }
    fclose(csv);
    printf("Done!\n\n");

    printf("===============================================================\n");
    printf("KEY FINDINGS\n");
    printf("===============================================================\n");

    if (idx1000 >= 0 && idx30 >= 0) {
        float avgE1000  = euler_time[idx1000] / euler_cnt[idx1000];
        float avgE30    = euler_time[idx30]   / euler_cnt[idx30];
        float avgAE1000 = ae_time[idx1000]    / ae_cnt[idx1000];
        float avgAE30   = ae_time[idx30]      / ae_cnt[idx30];

        printf("1. At M=1000: AE/Euler = %.2fx\n", avgAE1000 / avgE1000);
        printf("2. At M=30:   AE/Euler = %.2fx\n", avgAE30   / avgE30);
        printf("3. Using M=30 speeds up Euler by %.2fx, AE by %.2fx (vs M=1000)\n",
            avgE1000 / avgE30, avgAE1000 / avgAE30);
    } else {
        printf("Key findings require M=1000 and M=30 in M_values.\n");
    }
    printf("===============================================================\n");

    // Cleanup
    free(param_sets);
    free(results);
    free(h_partial_sums);
    testCUDA(cudaFree(d_payoffs));
    testCUDA(cudaFree(d_partial_sums));
    testCUDA(cudaFree(d_states_euler));
    testCUDA(cudaFree(d_states_almost));

    return 0;
}
