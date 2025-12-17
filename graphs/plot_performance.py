import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../results/benchmark_results.csv")

df["method"] = df["method"].astype(str).str.strip()
df = df[df["method"].isin(["Euler", "Almost Exact"])].copy()

df["dt"] = pd.to_numeric(df["dt"], errors="coerce")
df["time_ms"] = pd.to_numeric(df["time_ms"], errors="coerce")
df = df.dropna(subset=["dt", "time_ms"])

df["dt_round"] = df["dt"].round(12)

agg = (df.groupby(["method", "dt_round"], as_index=False)
         .agg(mean_time_ms=("time_ms", "mean"),
              std_time_ms=("time_ms", "std"),
              n=("time_ms", "size")))

dt_order = sorted(agg["dt_round"].unique(), reverse=True)
dt_to_x = {dt: i for i, dt in enumerate(dt_order)}
agg["x"] = agg["dt_round"].map(dt_to_x)

plt.figure(figsize=(9, 5.5))
for method, marker in [("Euler", "o"), ("Almost Exact", "s")]:
    sub = agg[agg["method"] == method].sort_values("x")
    plt.plot(sub["x"], sub["mean_time_ms"], marker=marker, label=method)

plt.xticks(range(len(dt_order)), [f"{dt:g}" for dt in dt_order], rotation=0)
plt.xlabel(r"$\Delta t$")
plt.ylabel("Mean runtime (ms)")
plt.title("Step 3: Runtime vs Δt (averaged over all parameter sets)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

print(agg.sort_values(["method", "dt_round"]))