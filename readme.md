# Heston Model – GPU Monte Carlo Pricing

[🎥 Demo – Exact Simulation Animation](results/demo_heston_2.mp4)

CUDA implementation of Monte Carlo pricing for a European call option under the
Heston stochastic volatility model.  
The project contains three main CUDA programs:

- `heston_1.cu` – Euler discretisation of \((S_t, v_t)\)
- `heston_2.cu` – (Almost) exact simulation of the variance process
- `heston_3.cu` – Performance benchmark over many parameter sets

The CUDA kernels are complemented by a small Python toolkit to analyse the
benchmarks and to animate a subset of simulated price paths.

---

## Mathematical model (short)

We consider the Heston model

\[
\begin{aligned}
dS_t &= r S_t \,dt + \sqrt{v_t}\, S_t \, d\widetilde W_t, \\
dv_t &= \kappa(\theta - v_t)\,dt + \sigma \sqrt{v_t}\, dW_t, \\
\widetilde W_t &= \rho W_t + \sqrt{1-\rho^2}\, Z_t,
\end{aligned}
\]

and price a European call

\[
C_0 = e^{-rT}\,\mathbb{E}\big[(S_T - K)^+\big].
\]

In the "almost exact" scheme the variance step uses the CIR
noncentral–chi–square / gamma representation

\[
v_{t+\Delta t}
=
\frac{\sigma^2 (1 - e^{-\kappa \Delta t})}{2\kappa}\;
\mathcal G(d + N),
\]

with \(N\sim\text{Poisson}(\lambda)\) and \(\mathcal G(\cdot)\) a gamma random
variable.

---

## Repository layout

```text
HestonModel_Finance/
├── README.md
├── docs/
│   └── subjects.pdf              # project assignment
├── src/
│   ├── compile.sh                # helper script to build all CUDA codes
│   ├── heston_1.cu               # Step 1 – Euler scheme
│   ├── heston_2.cu               # Step 2 – exact / almost exact variance
│   └── heston_3.cu               # Step 3 – benchmark
├── graphs/
│   ├── analyze_benchmark.py      # plots from benchmark_results.csv
│   └── animate_paths.py          # animation for selected price paths
└── results/
    ├── benchmark_results.csv
    ├── boxplot_time_by_method_M.png
    ├── scatter_time_vs_params.png
    ├── hist_price_diff.png
    ├── hist_rel_error.png
    ├── paths.csv
    ├── payoff_vs_time.png
    ├── demo_heston_2.mp4
    └── heston_paths_trading_style.mp4
```

---

## Requirements

### CUDA side

- NVIDIA GPU with compute capability ≥ sm_70
- CUDA toolkit (nvcc, curand)

### Python side (plots & animations)

- Python ≥ 3.8
- numpy, pandas, matplotlib
- ffmpeg available on the system path (for MP4 export)

Example environment:

```bash
conda create -n heston python=3.12
conda activate heston
pip install numpy pandas matplotlib
# install ffmpeg with your OS package manager
```

---

## Step 1 – Euler discretisation (`src/heston_1.cu`)

**Goal.** Basic Monte Carlo pricing using Euler steps for both \(S_t\) and
\(v_t\), with two ways of truncating the variance:

- \(g(x) = x^+ = \max(x, 0)\)
- \(g(x) = |x|\)

Each GPU thread simulates one independent path with

\[
S_{t+\Delta t} = S_t + r S_t \Delta t + \sqrt{v_t}\, S_t \sqrt{\Delta t}\,
(\rho G_1 + \sqrt{1-\rho^2}\, G_2),
\]

and writes the terminal payoff \((S_T - K)^+\) to global memory.

The code uses:

- a dedicated RNG initialisation kernel (curand),
- a standard shared–memory reduction to sum all payoffs per block,
- a final reduction on the CPU.

Typical configuration in the code:

- `THREADS_PER_BLOCK = 256`
- `NUM_BLOCKS = 1024`
- `TOTAL_PATHS = 262144`
- `M = 1000` time steps.

**Run:**

```bash
cd src
bash compile.sh    # or: nvcc -o heston_1 heston_1.cu -lcurand -arch=sm_70
./heston_1
```

---

## Step 2 – Exact / almost exact scheme (`src/heston_2.cu`)

**Goal.** Remove the bias of the Euler scheme on the variance by:

- Simulating \(v_t\) with the gamma–based exact law of the CIR process
  (Poisson + gamma),
- Using the closed–form representation of the integral
  \(\int_0^T \sqrt{v_s}\, dW_s\),
- Drawing \(S_T\) from a log–normal variable with parameters depending on
  \(v_0, v_T\) and \(\int_0^T v_s\, ds\).

Main ingredients:

- `__device__` function `gamma_distribution(curandState*, float alpha)` implementing
  Andersen's algorithm (non-recursive handling of alpha < 1);
- one kernel `heston_exact_kernel` that:
  - loops over time to update \(v_t\) exactly and accumulates
    \(v_I \approx \int_0^T v_s\, ds\),
  - computes the Brownian integral and the log-spot parameters,
  - samples \(S_T\) with a single Gaussian,
  - writes the payoff.

**Run:**

```bash
cd src
bash compile.sh    # or: nvcc -o heston_2 heston_2.cu -lcurand -arch=sm_70
./heston_2
```

The program prints the estimated option price and the GPU time for
`TOTAL_PATHS = 262144` and `M = 1000`.

---

## Step 3 – Performance benchmark (`src/heston_3.cu`)

**Goal.** Compare execution time and prices of:

- Euler scheme
- "Almost exact" scheme

over a grid of parameters:

- \(\kappa \in [0.1, 10]\),
- \(\theta \in [0.01, 0.5]\),
- \(\sigma \in [0.1, 1]\),
- \(\rho \in \{-0.7, -0.3, 0, 0.3, 0.7\}\),
- time steps \(M \in \{1000, 30\}\),

subject to the Feller condition \(2\kappa\theta > \sigma^2\).

For each test:

- Euler and Almost Exact are run with `TOTAL_PATHS = 262144` paths,
- GPU times are measured with CUDA events,
- results are stored in `results/benchmark_results.csv`.

**Run:**

```bash
cd src
bash compile.sh    # or: nvcc -o heston_3 heston_3.cu -lcurand -arch=sm_70
./heston_3
```

---

## Benchmark plots (`graphs/analyze_benchmark.py`)

To analyse the CSV produced by `heston_3`:

```bash
conda activate heston
python graphs/analyze_benchmark.py
```

This script generates several figures under `results/`, including

![Execution time vs parameters](results/scatter_time_vs_params.png)

*Figure – Execution time vs Heston parameters \(\kappa, \theta, \sigma\).*
*Each dot corresponds to one CUDA run with 262,144 simulated paths.*
*Colours distinguish Euler / Almost Exact and \(M \in \{30, 1000\}\).*

Other figures:

- `boxplot_time_by_method_M.png` – boxplot of timings grouped by method and \(M\)
- `hist_price_diff.png` – histogram of price differences
  (Almost Exact \(M = 30\) vs Euler \(M = 1000\))
- `hist_rel_error.png` – histogram of relative errors.
