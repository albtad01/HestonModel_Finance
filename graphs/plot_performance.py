import pandas as pd
import matplotlib.pyplot as plt

# 1) Load
df = pd.read_csv("../results/benchmark_results.csv")

# 2) Filter methods
df["method"] = df["method"].astype(str).str.strip()
df = df[df["method"].isin(["Euler", "Almost Exact"])].copy()

# 3) Numerics
df["dt"] = pd.to_numeric(df["dt"], errors="coerce")
df["time_ms"] = pd.to_numeric(df["time_ms"], errors="coerce")
df = df.dropna(subset=["dt", "time_ms"])

# 4) Group by dt (avoid float noise)
df["dt_round"] = df["dt"].round(12)

agg = (df.groupby(["method", "dt_round"], as_index=False)
         .agg(mean_time_ms=("time_ms", "mean"),
              std_time_ms=("time_ms", "std"),
              n=("time_ms", "size")))

# 5) Order dt from large -> small (so "finer dt" goes to the right)
dt_order = sorted(agg["dt_round"].unique(), reverse=True)

# Map dt -> categorical x positions (equally spaced)
dt_to_x = {dt: i for i, dt in enumerate(dt_order)}
agg["x"] = agg["dt_round"].map(dt_to_x)

# 6) Plot
plt.figure(figsize=(9, 5.5))
for method, marker in [("Euler", "o"), ("Almost Exact", "s")]:
    sub = agg[agg["method"] == method].sort_values("x")
    plt.plot(sub["x"], sub["mean_time_ms"], marker=marker, label=method)
    # Optional error bars (comment out if too busy)
    # plt.errorbar(sub["x"], sub["mean_time_ms"], yerr=sub["std_time_ms"],
    #              fmt="none", capsize=3)

plt.xticks(
    range(len(dt_order)),
    [f"{dt:g}" for dt in dt_order],
    rotation=0
)
plt.xlabel("dt")
plt.ylabel("Mean runtime (ms)")
plt.title("Step 3: Runtime vs dt (averaged over all parameter sets)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

print(agg.sort_values(["method", "dt_round"]))