import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = "../results/benchmark_results.csv"
V0 = 0.1

df = pd.read_csv(CSV_PATH)

df["method"] = df["method"].astype(str).str.strip()
df = df[df["method"].isin(["Euler", "Almost Exact"])].copy()

for c in ["kappa", "theta", "sigma", "dt", "time_ms"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(subset=["kappa", "theta", "sigma", "dt", "time_ms"])

sig2 = df["sigma"] ** 2
df["d"] = 2.0 * df["kappa"] * df["theta"] / sig2

exp_kdt = np.exp(-df["kappa"] * df["dt"])
den = sig2 * (1.0 - exp_kdt)
df["lambda"] = (2.0 * df["kappa"] * exp_kdt * V0) / den

df["d_r"] = df["d"].round(12)
df["lambda_r"] = df["lambda"].round(12)

agg_d = (df.groupby(["method", "d_r"], as_index=False)
           .agg(mean_time_ms=("time_ms", "mean"))
           .sort_values(["method", "d_r"]))

agg_l = (df.groupby(["method", "lambda_r"], as_index=False)
           .agg(mean_time_ms=("time_ms", "mean"))
           .sort_values(["method", "lambda_r"]))

plt.figure(figsize=(9, 5.5))
for method, marker in [("Euler", "o"), ("Almost Exact", "s")]:
    sub = agg_l[agg_l["method"] == method].sort_values("lambda_r")
    plt.plot(sub["lambda_r"], sub["mean_time_ms"], marker=marker, label=method)
plt.xlabel(r"$\lambda$")
plt.ylabel("Mean runtime (ms)")
plt.title(r"Runtime vs $\lambda$")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(9, 5.5))
for method, marker in [("Euler", "o"), ("Almost Exact", "s")]:
    sub = agg_d[agg_d["method"] == method].sort_values("d_r")
    plt.plot(sub["d_r"], sub["mean_time_ms"], marker=marker, label=method)
plt.xscale("log")
plt.xlabel(r"$d$ (log scale)")
plt.ylabel("Mean runtime (ms)")
plt.title(r"Runtime vs $d$")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()