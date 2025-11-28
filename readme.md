# Heston Model – GPU Monte Carlo Pricing

[🎥 Demo – Exact Simulation Animation](results/demo_heston_2.mp4)

GPU‐accelerated Monte Carlo pricing of a European call option under the Heston
stochastic volatility model, implemented in CUDA and benchmarked on modern
NVIDIA GPUs.

We compare:

- **Euler discretization** of \((S_t, v_t)\)
- **“Almost exact” variance scheme** based on the noncentral chi–square / gamma
  representation of the CIR process
- Performance & accuracy for different time steps \(M\) and parameter regimes

---

## Mathematical model

We consider the Heston model

\[
\begin{aligned}
dS_t &= r S_t \,dt + \sqrt{v_t}\, S_t \, d\widetilde W_t, \\
dv_t &= \kappa(\theta - v_t)\,dt + \sigma \sqrt{v_t}\, dW_t, \\
\widetilde W_t &= \rho W_t + \sqrt{1-\rho^2}\, Z_t,
\end{aligned}
\]

and price the **European call**

\[
C_0 = e^{-rT}\,\mathbb{E}\big[(S_T - K)^+\big].
\]

For the almost–exact variance scheme we use

\[
v_{t+\Delta t}
=
\frac{\sigma^2 (1 - e^{-\kappa \Delta t})}{2\kappa}\;
\mathcal G(d + N),
\]

where \(N\sim\text{Poisson}(\lambda)\) and \(\mathcal G(\cdot)\) is a gamma random
variable as in Andersen (2008).

---

## Repository structure

```text
HestonModel_Finance/
├── README.md
├── docs/
│   └── subjects.pdf              # project description from the course
├── src/
│   ├── compile.sh                # helper script to build CUDA executables
│   ├── heston_1.cu               # Step 1 – Euler discretization
│   ├── heston_2.cu               # Step 2 – Exact / almost-exact variance
│   └── heston_3.cu               # Step 3 – Benchmark over parameter sets
├── graphs/
│   ├── analyze_benchmark.py      # boxplots, scatter plots, histograms
│   └── animate_paths.py          # fancy MP4 animation of simulated paths
└── results/
    ├── benchmark_results.csv
    ├── boxplot_time_by_method_M.png
    ├── hist_price_diff.png
    ├── hist_rel_error.png
    ├── scatter_time_vs_params.png
    ├── paths.csv                 # saved paths for visualization
    ├── payoff_vs_time.png
    ├── demo_heston_2.mp4         # short demo video for Step 2
    └── heston_paths_trading_style.mp4  # animated paths + mean S(t)