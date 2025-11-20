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
#define NUM_BLOCKS 1024
#define TOTAL_PATHS (THREADS_PER_BLOCK * NUM_BLOCKS)

// Fixed parameters
#define S0 1.0f
#define v0 0.1f
#define r 0.0f
#define T 1.0f
#define K 1.0f

void testCUDA(cudaError_t error, const char *file, int line) {
    if (error != cudaSuccess) {
        printf("CUDA error at %s:%d: %s\n", file, line, cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}
#define testCUDA(error) (testCUDA(error, __FILE__, __LINE__))


/////////////////////////////////////////////////////////////////////////////
// STRUCTS
/////////////////////////////////////////////////////////////////////////////
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
    int M;
    const char* method_name;
    float time_ms;
    float price;
} BenchmarkResult;


/////////////////////////////////////////////////////////////////////////////
// GAMMA DISTRIBUTION (from Marsaglia & Tsang 2000)
/////////////////////////////////////////////////////////////////////////////
__device__ float gamma_distribution(curandState *state, float alpha) {
    
    // Handle α < 1 first (avoid recursion)
    float boost_factor = 1.0f;
    if (alpha < 1.0f) {
        boost_factor = powf(curand_uniform(state), 1.0f / alpha);
        alpha = alpha + 1.0f;
    }
    
    // Now α ≥ 1: Marsaglia & Tsang algorithm
    float d = alpha - 1.0f/3.0f;
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


/////////////////////////////////////////////////////////////////////////////
// KERNEL 1: EULER DISCRETIZATION
/////////////////////////////////////////////////////////////////////////////
__global__ void heston_euler_kernel(
    float *payoffs, 
    unsigned long seed,
    float kappa,
    float theta,
    float sigma,
    float rho,
    int M) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    curandState state;
    curand_init(seed, idx, 0, &state);
    
    float dt = T / (float)M;
    float sqrt_dt = sqrtf(dt);
    float sqrt_1_rho2 = sqrtf(1.0f - rho * rho);
    
    float S = S0;
    float v = v0;
    
    // Euler time-stepping
    for (int step = 0; step < M; step++) {
        
        float G1 = curand_normal(&state);
        float G2 = curand_normal(&state);
        float dZ = rho * G1 + sqrt_1_rho2 * G2;
        
        // Update S
        S = S + r * S * dt + sqrtf(fmaxf(v, 0.0f)) * S * sqrt_dt * dZ;
        
        // Update v
        float v_new = v + kappa * (theta - v) * dt 
                        + sigma * sqrtf(fmaxf(v, 0.0f)) * sqrt_dt * G1;
        v = fmaxf(v_new, 0.0f);
    }
    
    payoffs[idx] = fmaxf(S - K, 0.0f);
}


/////////////////////////////////////////////////////////////////////////////
// KERNEL 2: ALMOST EXACT SCHEME
/////////////////////////////////////////////////////////////////////////////
__global__ void heston_almost_exact_kernel(
    float *payoffs,
    unsigned long seed,
    float kappa,
    float theta,
    float sigma,
    float rho,
    int M) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    curandState state;
    curand_init(seed, idx, 0, &state);
    
    float dt = T / (float)M;
    float sqrt_dt = sqrtf(dt);
    float exp_kappa_dt = expf(-kappa * dt);
    
    // Precompute constants for variance
    float d = 2.0f * kappa * theta / (sigma * sigma);
    float coeff = sigma * sigma * (1.0f - exp_kappa_dt) / (2.0f * kappa);
    
    // Precompute k0, k1, k2 for almost exact scheme
    float k0 = (-rho / sigma * kappa * theta) * dt;
    float k1 = (rho * kappa / sigma - 0.5f) * dt - rho / sigma;
    float k2 = rho / sigma;
    
    float sqrt_1_rho2 = sqrtf(1.0f - rho * rho);
    
    // Initialize
    float log_S = logf(S0);
    float v = v0;
    
    // Time loop
    for (int step = 0; step < M; step++) {
        
        float v_old = v;
        
        // Simulate v_{t+Δt} using exact distribution
        float lambda = 2.0f * kappa * exp_kappa_dt * v_old / 
                       (sigma * sigma * (1.0f - exp_kappa_dt));
        
        unsigned int N = curand_poisson(&state, lambda);
        float alpha = 0.5f * d + (float)N;
        float gamma_sample = gamma_distribution(&state, alpha);
        
        v = coeff * gamma_sample;
        
        // Generate independent normals
        float G1 = curand_normal(&state);
        float G2 = curand_normal(&state);
        
        // Almost exact update for log(S)
        log_S = log_S + k0 + k1 * v_old + k2 * v 
                + sqrtf((1.0f - rho * rho) * v_old * dt) * (rho * G1 + sqrt_1_rho2 * G2);
    }
    
    float S = expf(log_S);
    payoffs[idx] = fmaxf(S - K, 0.0f);
}


/////////////////////////////////////////////////////////////////////////////
// REDUCTION KERNEL
/////////////////////////////////////////////////////////////////////////////
__global__ void reduction_kernel(float *payoffs, float *partial_sums, int N) {
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


/////////////////////////////////////////////////////////////////////////////
// BENCHMARK FUNCTION
/////////////////////////////////////////////////////////////////////////////
BenchmarkResult run_benchmark(
    float kappa, float theta, float sigma, float rho, int M,
    bool use_almost_exact, const char* method_name) {
    
    float *d_payoffs, *d_partial_sums;
    testCUDA(cudaMalloc(&d_payoffs, TOTAL_PATHS * sizeof(float)));
    testCUDA(cudaMalloc(&d_partial_sums, NUM_BLOCKS * sizeof(float)));
    
    cudaEvent_t start, stop;
    testCUDA(cudaEventCreate(&start));
    testCUDA(cudaEventCreate(&stop));
    testCUDA(cudaEventRecord(start, 0));
    
    // Launch appropriate kernel
    if (use_almost_exact) {
        heston_almost_exact_kernel<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(
            d_payoffs, 12345UL, kappa, theta, sigma, rho, M);
    } else {
        heston_euler_kernel<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(
            d_payoffs, 12345UL, kappa, theta, sigma, rho, M);
    }
    testCUDA(cudaGetLastError());
    
    // Reduction
    reduction_kernel<<<NUM_BLOCKS, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(float)>>>(
        d_payoffs, d_partial_sums, TOTAL_PATHS);
    testCUDA(cudaGetLastError());
    
    testCUDA(cudaEventRecord(stop, 0));
    testCUDA(cudaEventSynchronize(stop));
    
    // Copy and compute price
    float *h_partial_sums = (float*)malloc(NUM_BLOCKS * sizeof(float));
    testCUDA(cudaMemcpy(h_partial_sums, d_partial_sums, NUM_BLOCKS * sizeof(float), 
                        cudaMemcpyDeviceToHost));
    
    float total = 0.0f;
    for (int i = 0; i < NUM_BLOCKS; i++) {
        total += h_partial_sums[i];
    }
    float price = total / TOTAL_PATHS;
    
    float time_ms;
    testCUDA(cudaEventElapsedTime(&time_ms, start, stop));
    
    free(h_partial_sums);
    testCUDA(cudaFree(d_payoffs));
    testCUDA(cudaFree(d_partial_sums));
    testCUDA(cudaEventDestroy(start));
    testCUDA(cudaEventDestroy(stop));
    
    BenchmarkResult result;
    result.kappa = kappa;
    result.theta = theta;
    result.sigma = sigma;
    result.rho = rho;
    result.M = M;
    result.method_name = method_name;
    result.time_ms = time_ms;
    result.price = price;
    
    return result;
}


/////////////////////////////////////////////////////////////////////////////
// PARAMETER GENERATION
/////////////////////////////////////////////////////////////////////////////
void generate_param_sets(ParamSet *params, int n_samples) {
    srand(time(NULL));
    
    int count = 0;
    int attempts = 0;
    const int max_attempts = n_samples * 100;
    
    while (count < n_samples && attempts < max_attempts) {
        attempts++;
        
        // Random sampling
        float kappa = 0.1f + (float)rand() / RAND_MAX * (10.0f - 0.1f);
        float theta = 0.01f + (float)rand() / RAND_MAX * (0.5f - 0.01f);
        float sigma = 0.1f + (float)rand() / RAND_MAX * (1.0f - 0.1f);
        
        // Check Feller condition: 2κθ > σ²
        if (2.0f * kappa * theta > sigma * sigma) {
            params[count].kappa = kappa;
            params[count].theta = theta;
            params[count].sigma = sigma;
            count++;
        }
    }
    
    if (count < n_samples) {
        printf("Warning: Only generated %d/%d valid parameter sets\n", count, n_samples);
    }
}


/////////////////////////////////////////////////////////////////////////////
// MAIN
/////////////////////////////////////////////////////////////////////////////
int main(void) {
    
    printf("===============================================================\n");
    printf("Heston Model Monte Carlo - Step 3: Performance Comparison\n");
    printf("===============================================================\n");
    printf("Testing: κ ∈ [0.1, 10], θ ∈ [0.01, 0.5], σ ∈ [0.1, 1]\n");
    printf("         ρ ∈ {-0.7, -0.3, 0, 0.3, 0.7}\n");
    printf("         Δt ∈ {1/1000, 1/30}\n");
    printf("Constraint: 2κθ > σ² (Feller condition)\n");
    printf("Paths per test: %d\n", TOTAL_PATHS);
    printf("===============================================================\n\n");
    
    // Parameter sets to test
    const int N_PARAM_SETS = 30;
    ParamSet *param_sets = (ParamSet*)malloc(N_PARAM_SETS * sizeof(ParamSet));
    
    printf("Generating %d random parameter sets (κ, θ, σ)...\n", N_PARAM_SETS);
    generate_param_sets(param_sets, N_PARAM_SETS);
    
    // Print generated parameters
    printf("\nGenerated parameter sets:\n");
    printf("%-4s %-8s %-8s %-8s %-12s\n", "ID", "κ", "θ", "σ", "2κθ/σ²");
    printf("---------------------------------------------------\n");
    for (int i = 0; i < N_PARAM_SETS; i++) {
        float k = param_sets[i].kappa;
        float t = param_sets[i].theta;
        float s = param_sets[i].sigma;
        printf("%-4d %-8.3f %-8.4f %-8.3f %-12.3f\n", 
               i+1, k, t, s, (2*k*t)/(s*s));
    }
    printf("\n");
    
    // Rho values to test
    float rho_values[] = {-0.7f, -0.3f, 0.0f, 0.3f, 0.7f};
    int n_rho = 5;
    
    // Timesteps to test
    int M_values[] = {1000, 30};
    int n_M = 2;
    
    // Results storage
    int total_tests = N_PARAM_SETS * n_rho * n_M * 2;
    BenchmarkResult *results = (BenchmarkResult*)malloc(total_tests * sizeof(BenchmarkResult));
    
    printf("Running %d benchmarks (this may take several minutes)...\n", total_tests);
    printf("Progress: [");
    fflush(stdout);
    
    int result_idx = 0;
    int progress_step = total_tests / 50;
    
    // Main benchmark loop - VARIABILI RINOMINATE!
    for (int p = 0; p < N_PARAM_SETS; p++) {
        float kappa = param_sets[p].kappa;
        float theta = param_sets[p].theta;
        float sigma = param_sets[p].sigma;
        
        for (int rho_idx = 0; rho_idx < n_rho; rho_idx++) {  // ← rho_idx invece di r
            float rho = rho_values[rho_idx];
            
            for (int m_idx = 0; m_idx < n_M; m_idx++) {      // ← m_idx invece di m
                int M = M_values[m_idx];
                
                // Euler
                results[result_idx++] = run_benchmark(
                    kappa, theta, sigma, rho, M, false, "Euler");
                
                // Almost Exact
                results[result_idx++] = run_benchmark(
                    kappa, theta, sigma, rho, M, true, "Almost Exact");
                
                // Progress bar
                if ((result_idx % progress_step) == 0) {
                    printf("=");
                    fflush(stdout);
                }
            }
        }
    }
    printf("] Done!\n\n");
    
    // ===================================================================
    // ANALYSIS
    // ===================================================================
    
    printf("===============================================================\n");
    printf("PERFORMANCE ANALYSIS\n");
    printf("===============================================================\n\n");
    
    // Analysis by Δt
    printf("1. COMPARISON BY TIME STEP\n");
    printf("-----------------------------------------------------------\n");
    
    float euler_time_1000 = 0, almost_time_1000 = 0;
    float euler_time_30 = 0, almost_time_30 = 0;
    int count_1000 = 0, count_30 = 0;
    
    for (int i = 0; i < total_tests; i++) {
        if (results[i].M == 1000) {
            if (strcmp(results[i].method_name, "Euler") == 0) {
                euler_time_1000 += results[i].time_ms;
                count_1000++;
            } else {
                almost_time_1000 += results[i].time_ms;
            }
        } else {
            if (strcmp(results[i].method_name, "Euler") == 0) {
                euler_time_30 += results[i].time_ms;
                count_30++;
            } else {
                almost_time_30 += results[i].time_ms;
            }
        }
    }
    
    printf("Δt = 1/1000 (M = 1000 steps):\n");
    printf("  Euler:        %.2f ms (avg over %d tests)\n", 
           euler_time_1000 / count_1000, count_1000);
    printf("  Almost Exact: %.2f ms (avg over %d tests)\n", 
           almost_time_1000 / count_1000, count_1000);
    printf("  Ratio:        %.3fx (Almost/Euler)\n\n", 
           almost_time_1000 / euler_time_1000);
    
    printf("Δt = 1/30 (M = 30 steps):\n");
    printf("  Euler:        %.2f ms (avg over %d tests)\n", 
           euler_time_30 / count_30, count_30);
    printf("  Almost Exact: %.2f ms (avg over %d tests)\n", 
           almost_time_30 / count_30, count_30);
    printf("  Ratio:        %.3fx (Almost/Euler)\n\n", 
           almost_time_30 / euler_time_30);
    
    printf("Impact of using Δt = 1/30:\n");
    printf("  Euler speedup:        %.2fx faster\n",
           (euler_time_1000/count_1000) / (euler_time_30/count_30));
    printf("  Almost Exact speedup: %.2fx faster\n\n",
           (almost_time_1000/count_1000) / (almost_time_30/count_30));
    
    // Save CSV
    printf("Saving results to benchmark_results.csv...\n");
    FILE *csv = fopen("benchmark_results.csv", "w");
    fprintf(csv, "test_id,kappa,theta,sigma,rho,M,dt,method,time_ms,price\n");
    for (int i = 0; i < total_tests; i++) {
        BenchmarkResult res = results[i];  // ← res invece di r
        fprintf(csv, "%d,%.6f,%.6f,%.6f,%.6f,%d,%.6f,%s,%.3f,%.6f\n",
                i+1, res.kappa, res.theta, res.sigma, res.rho, res.M, T/(float)res.M,
                res.method_name, res.time_ms, res.price);
    }
    fclose(csv);
    printf("Done!\n\n");
    
    printf("===============================================================\n");
    printf("KEY FINDINGS\n");
    printf("===============================================================\n");
    printf("1. Almost Exact is ~%.1fx slower than Euler for Δt=1/1000\n",
           almost_time_1000 / euler_time_1000);
    printf("2. Almost Exact is ~%.1fx slower than Euler for Δt=1/30\n",
           almost_time_30 / euler_time_30);
    printf("3. Using Δt=1/30 speeds up simulations significantly\n");
    printf("4. Trade-off: Almost Exact is slower but more accurate!\n");
    printf("===============================================================\n");
    
    free(param_sets);
    free(results);
    
    return 0;
}