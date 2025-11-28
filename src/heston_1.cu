// Alberto Taddei & Thies Weel
// Heston Model Monte Carlo Simulation - Step 1: Euler Discretization

#include <stdio.h>
#include <math.h>
#include <curand_kernel.h>

#define THREADS_PER_BLOCK 256
#define NUM_BLOCKS 1024
#define TOTAL_PATHS (THREADS_PER_BLOCK * NUM_BLOCKS)  // 262,144 paths

// Model parameters
#define S0 1.0f
#define v0 0.1f
#define r 0.0f
#define kappa 0.5f
#define theta 0.1f
#define sigma 0.3f
#define rho 0.0f
#define T 1.0f
#define K 1.0f
#define M 1000  // Number of time steps (Δt = 1/M)

// Function to catch CUDA errors
void testCUDA(cudaError_t error, const char *file, int line) {
    if (error != cudaSuccess) {
        printf("CUDA error at %s:%d: %s\n", file, line, cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}
#define testCUDA(error) (testCUDA(error, __FILE__, __LINE__))

// Random Number Generator init kernel
__global__ void init_rng_kernel(curandState *states, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < TOTAL_PATHS) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

// Euler Discretization Kernel
// Simulates one path of (S_t, v_t) using Euler scheme and returns payoff
__global__ void heston_euler_kernel(float *payoffs, curandState *states, int use_abs) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    curandState localState = states[idx];
    
    // Time discretization
    float dt = T / (float)M;
    float sqrt_dt = sqrtf(dt);
    float sqrt_1_minus_rho2 = sqrtf(1.0f - rho * rho);
    
    // Initialize state variables
    float S = S0;
    float v = v0;
    
    // Time loop: Euler discretization
    for (int step = 0; step < M; step++) {
        
        // Generate two independent standard normal random variables
        float G1 = curand_normal(&localState);
        float G2 = curand_normal(&localState);
        
        // Compute correlated Brownian motion for S
        float dZ = rho * G1 + sqrt_1_minus_rho2 * G2;
        
        // Update asset price S using equation (4)
        // S_{t+Δt} = S_t + r*S_t*Δt + √v_t * S_t * √Δt * dZ
        S = S + r * S * dt + sqrtf(fmaxf(v, 0.0f)) * S * sqrt_dt * dZ;
        
        // Update variance v using equation (5)
        // v_{t+Δt} = g(v_t + κ(θ-v_t)Δt + σ√v_t√Δt*G1)
        float v_new = v + kappa * (theta - v) * dt + sigma * sqrtf(fmaxf(v, 0.0f)) * sqrt_dt * G1;
        
        // Apply function g: either (·)+ or |·|
        if (use_abs) {
            v = fabsf(v_new); // |·|
        } else {
            v = fmaxf(v_new, 0.0f);  // (·)+ = max(·, 0)
        }
    }
    
    // Compute payoff: (S_T - K)^+ = max(S_T - K, 0)
    float payoff = fmaxf(S - K, 0.0f);
    
    // Store result
    payoffs[idx] = payoff;
    states[idx] = localState; 

}

// Reduction kernel to sum payoffs (parallel reduction)
__global__ void reduction_kernel(float *payoffs, float *partial_sums, int N) {
    
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data into shared memory
    sdata[tid] = (idx < N) ? payoffs[idx] : 0.0f;
    __syncthreads();
    
    // Parallel reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Write result for this block to global memory
    if (tid == 0) {
        partial_sums[blockIdx.x] = sdata[tid];
    }
}

// Host function to run Euler simulation
float heston_euler_simulation(int use_abs, int verbose = 1) {
    
    // Allocate device memory
    float *d_payoffs, *d_partial_sums;
    curandState *d_states;
    testCUDA(cudaMalloc(&d_payoffs, TOTAL_PATHS * sizeof(float)));
    testCUDA(cudaMalloc(&d_partial_sums, NUM_BLOCKS * sizeof(float)));
    testCUDA(cudaMalloc(&d_states, TOTAL_PATHS * sizeof(curandState)));
    
    // 1) Random Number Generator initialization
    unsigned long seed = 12345UL;
    init_rng_kernel<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_states, seed);
    testCUDA(cudaGetLastError());
    testCUDA(cudaDeviceSynchronize());

    // 2) Timing
    cudaEvent_t start, stop;
    testCUDA(cudaEventCreate(&start));
    testCUDA(cudaEventCreate(&stop));
    testCUDA(cudaEventRecord(start, 0));

    // 3) Launch Euler kernel
    heston_euler_kernel<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_payoffs, d_states, use_abs);
    testCUDA(cudaGetLastError());
    
    // Reduction to sum all payoffs
    reduction_kernel<<<NUM_BLOCKS, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(float)>>>(
        d_payoffs, d_partial_sums, TOTAL_PATHS);
    testCUDA(cudaGetLastError());
    
    // Copy partial sums to host and finish reduction on CPU
    float *h_partial_sums = (float*)malloc(NUM_BLOCKS * sizeof(float));
    testCUDA(cudaMemcpy(h_partial_sums, d_partial_sums, NUM_BLOCKS * sizeof(float), 
                        cudaMemcpyDeviceToHost));
    
    float total_sum = 0.0f;
    for (int i = 0; i < NUM_BLOCKS; i++) {
        total_sum += h_partial_sums[i];
    }
    
    float option_price = total_sum / TOTAL_PATHS;
    
    // Timing
    testCUDA(cudaEventRecord(stop, 0));
    testCUDA(cudaEventSynchronize(stop));
    float elapsed_time;
    testCUDA(cudaEventElapsedTime(&elapsed_time, start, stop));
    
    if (verbose) {
        printf("\n=== Euler Discretization (g = %s) ===\n", use_abs ? "|·|" : "(·)+");
        printf("Parameters: κ=%.2f, θ=%.2f, σ=%.2f, ρ=%.2f\n", kappa, theta, sigma, rho);
        printf("Paths: %d, Time steps: %d (Δt = %.6f)\n", TOTAL_PATHS, M, T/(float)M);
        printf("Estimated option price E[(S_1 - 1)+] = %.6f\n", option_price);
        printf("Execution time: %.3f ms\n", elapsed_time);
    }
    
    // Cleanup
    free(h_partial_sums);
    testCUDA(cudaFree(d_payoffs));
    testCUDA(cudaFree(d_partial_sums));
    testCUDA(cudaFree(d_states));
    testCUDA(cudaEventDestroy(start));
    testCUDA(cudaEventDestroy(stop));
    
    return option_price;
}

// Main function
int main(void) {
    
    printf("=============================================================\n");
    printf("Heston Model Monte Carlo Simulation - Step 1: Euler Scheme\n");
    printf("=============================================================\n");
    printf("Model: dS_t = rS_t dt + √v_t S_t dẐ_t\n");
    printf("       dv_t = κ(θ - v_t)dt + σ√v_t dW_t\n");
    printf("       Ẑ_t = ρW_t + √(1-ρ²)Z_t\n");
    printf("=============================================================\n");
    printf("Initial values: S_0 = %.2f, v_0 = %.2f\n", S0, v0);
    printf("Parameters: r = %.2f, K = %.2f, T = %.2f\n", r, K, T);
    printf("=============================================================\n");
    
    // Test with g(x) = (x)+
    float price_positive = heston_euler_simulation(0);  // use_abs = 0 : (·)+
    
    // Test with g(x) = |x|
    float price_abs = heston_euler_simulation(1);  // use_abs = 1 : |·|
    
    printf("\n=============================================================\n");
    printf("Comparison of variance truncation methods:\n");
    printf("  g(x) = (x)+  : Option price = %.6f\n", price_positive);
    printf("  g(x) = |x|   : Option price = %.6f\n", price_abs);
    printf("  Difference   : %.6f\n", fabsf(price_positive - price_abs));
    printf("=============================================================\n");
    
    return 0;
}