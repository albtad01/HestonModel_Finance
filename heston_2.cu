// Alberto Taddei & Thies Weel
// Heston Model Monte Carlo - Step 2: Exact Simulation with Gamma Distribution

#include <stdio.h>
#include <math.h>
#include <curand_kernel.h>

#define THREADS_PER_BLOCK 256
#define NUM_BLOCKS 1024
#define TOTAL_PATHS (THREADS_PER_BLOCK * NUM_BLOCKS)

// Model parameters
#define S0 1.0f
#define v0 0.1f
#define r 0.0f
#define kappa 0.5f
#define theta 0.1f
#define sigma 0.3f
#define rho -0.5f  // (can be changed)
#define T 1.0f
#define K 1.0f
#define M 1000

void testCUDA(cudaError_t error, const char *file, int line) {
    if (error != cudaSuccess) {
        printf("CUDA error at %s:%d: %s\n", file, line, cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}
#define testCUDA(error) (testCUDA(error, __FILE__, __LINE__))


/////////////////////////////////////////////////////////////////////////////
// GAMMA DISTRIBUTION DEVICE FUNCTION - Paper [8]
/////////////////////////////////////////////////////////////////////////////
__device__ float gamma_distribution(curandState *state, float alpha) {
    
    if (alpha >= 1.0f) {
        // Case α >= 1: algorithm [8]
        float d = alpha - 1.0f/3.0f;
        float c = 1.0f / sqrtf(9.0f * d);
        
        while (true) {
            float x, v;
            
            do {
                x = curand_normal(state);
                v = 1.0f + c * x;
            } while (v <= 0.0f);
            
            v = v * v * v;  // v = (1 + cx)³
            float u = curand_uniform(state);
            
            float x2 = x * x;
            if (u < 1.0f - 0.0331f * x2 * x2) {
                return d * v;
            }
            
            if (logf(u) < 0.5f * x2 + d * (1.0f - v + logf(v))) {
                return d * v;
            }
        }
        
    } else {
        // Case α < 1: Use Gamma(α+1) and scale
        // Gamma(α) = Gamma(α+1) × U^(1/α)
        float gamma_plus_1 = gamma_distribution(state, alpha + 1.0f);
        float u = curand_uniform(state);
        return gamma_plus_1 * powf(u, 1.0f / alpha);
    }
}


/////////////////////////////////////////////////////////////////////////////
// STEP 2: Exact Simulation Kernel
// Uses exact distribution for v_t and stochastic integral for S_t
/////////////////////////////////////////////////////////////////////////////
__global__ void heston_exact_kernel(float *payoffs, unsigned long seed) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    curandState state;
    curand_init(seed, idx, 0, &state);
    
    float dt = T / (float)M;
    float exp_kappa_dt = expf(-kappa * dt);
    
    // Precompute constants for variance simulation
    float d = 2.0f * kappa * theta / (sigma * sigma);  // degrees of freedom
    float coeff = sigma * sigma * (1.0f - exp_kappa_dt) / (2.0f * kappa);
    
    // Initialize
    float v = v0;
    float vI = 0.0f; 
    
    // Time loop: simulate v_t using exact distribution
    for (int step = 0; step < M; step++) {
        
        float v_old = v;
        
        // Step 1: Simulate v_{t+Δt} using exact distribution
        float lambda = 2.0f * kappa * exp_kappa_dt * v_old / (sigma * sigma * (1.0f - exp_kappa_dt));
        
        // N ~ Poisson(λ)
        unsigned int N = curand_poisson(&state, lambda);
        
        float alpha = 0.5f * d + (float)N;  // Shape parameter
        float gamma_sample = gamma_distribution(&state, alpha);
        
        v = coeff * gamma_sample;
        
        vI += 0.5f * (v_old + v) * dt;
    }
    // ∫₀¹ √vₛ dWₛ = (v₁ - v₀ - κθ + κvI) / σ
    float integral_sqrt_v_dW = (v - v0 - kappa * theta + kappa * vI) / sigma;
    
    // Compute m and sigma²
    float m = -0.5f * vI + rho * integral_sqrt_v_dW;
    float Sigma2 = (1.0f - rho * rho) * vI;
    
    // Sample S₁ = exp(m + Σ·G) where G is N(0,1)
    float G = curand_normal(&state);
    float S1 = S0 * expf(m + sqrtf(Sigma2) * G);
    
    // Payoff
    payoffs[idx] = fmaxf(S1 - K, 0.0f);
}


/////////////////////////////////////////////////////////////////////////////
// Reduction kernel (same as before)
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
// Host function for exact simulation
/////////////////////////////////////////////////////////////////////////////
float heston_exact_simulation() {
    
    float *d_payoffs, *d_partial_sums;
    testCUDA(cudaMalloc(&d_payoffs, TOTAL_PATHS * sizeof(float)));
    testCUDA(cudaMalloc(&d_partial_sums, NUM_BLOCKS * sizeof(float)));
    
    cudaEvent_t start, stop;
    testCUDA(cudaEventCreate(&start));
    testCUDA(cudaEventCreate(&stop));
    testCUDA(cudaEventRecord(start, 0));
    
    // Launch exact simulation kernel
    heston_exact_kernel<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_payoffs, 12345UL);
    testCUDA(cudaGetLastError());
    
    // Reduction
    reduction_kernel<<<NUM_BLOCKS, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(float)>>>(
        d_payoffs, d_partial_sums, TOTAL_PATHS);
    testCUDA(cudaGetLastError());
    
    // Copy and sum on CPU
    float *h_partial_sums = (float*)malloc(NUM_BLOCKS * sizeof(float));
    testCUDA(cudaMemcpy(h_partial_sums, d_partial_sums, NUM_BLOCKS * sizeof(float), 
                        cudaMemcpyDeviceToHost));
    
    float total = 0.0f;
    for (int i = 0; i < NUM_BLOCKS; i++) {
        total += h_partial_sums[i];
    }
    
    float option_price = total / TOTAL_PATHS;
    
    testCUDA(cudaEventRecord(stop, 0));
    testCUDA(cudaEventSynchronize(stop));
    float time_ms;
    testCUDA(cudaEventElapsedTime(&time_ms, start, stop));
    
    printf("\n=== Exact Simulation Results ===\n");
    printf("Method: Exact variance distribution + Poisson + Gamma\n");
    printf("Parameters: κ=%.2f, θ=%.2f, σ=%.2f, ρ=%.2f\n", kappa, theta, sigma, rho);
    printf("Paths: %d, Time steps: %d (Δt=%.6f)\n", TOTAL_PATHS, M, T/(float)M);
    printf("Option price E[(S₁ - 1)₊] = %.6f\n", option_price);
    printf("Execution time: %.3f ms (%.2f M paths/sec)\n", 
           time_ms, TOTAL_PATHS / (time_ms * 1000.0f));
    printf("================================\n");
    
    free(h_partial_sums);
    testCUDA(cudaFree(d_payoffs));
    testCUDA(cudaFree(d_partial_sums));
    testCUDA(cudaEventDestroy(start));
    testCUDA(cudaEventDestroy(stop));
    
    return option_price;
}


/////////////////////////////////////////////////////////////////////////////
// Main
/////////////////////////////////////////////////////////////////////////////
int main(void) {
    
    printf("============================================================\n");
    printf("Heston Model Monte Carlo - Step 2: Exact Simulation\n");
    printf("============================================================\n");
    printf("Using exact distribution for variance:\n");
    printf("  v_{t+Δt} ~ σ²(1-e^{-κΔt})/(2κ) × χ²(d+2N, λ)\n");
    printf("  N ~ Poisson(λ), χ² simulated via Gamma distribution\n");
    printf("============================================================\n");
    
    float price = heston_exact_simulation();
    
    printf("\nVerification: Feller condition 2κθ > σ²\n");
    printf("  2κθ = %.4f, σ² = %.4f → %s\n", 
           2*kappa*theta, sigma*sigma,
           (2*kappa*theta > sigma*sigma) ? "✓ Satisfied" : "✗ NOT satisfied");
    
    return 0;
}