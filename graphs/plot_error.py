import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# Config
# ----------------------------
CSV_PATH = "../results/benchmark_results.csv"

M_REF = 100000                       # reference Almost Exact M=1e5
HIST_M = 30                          # which Euler M to use for histogram (change if you want)
OUT_ERR_PDF  = "fig_error_vs_dt_ref_AE_M1e5.pdf"
OUT_HIST_PDF = "fig_hist_deltaP_Euler_minus_AE_ref_M1e5_M30.pdf"

# ----------------------------
# Load + clean
# ----------------------------
df = pd.read_csv(CSV_PATH)

df["method"]  = df["method"].astype(str).str.strip()
df["M"]       = pd.to_numeric(df["M"], errors="coerce")
df["dt"]      = pd.to_numeric(df["dt"], errors="coerce")
df["time_ms"] = pd.to_numeric(df["time_ms"], errors="coerce")
df["price"]   = pd.to_numeric(df["price"], errors="coerce")

df = df.dropna(subset=["kappa","theta","sigma","rho","M","dt","price","method"])

# Use rounded keys to avoid float matching issues (CSV prints 6 decimals)
for c in ["kappa", "theta", "sigma", "rho"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(subset=["kappa","theta","sigma","rho"])

df["kappa_r"] = df["kappa"].round(6)
df["theta_r"] = df["theta"].round(6)
df["sigma_r"] = df["sigma"].round(6)
df["rho_r"]   = df["rho"].round(6)

KEYS = ["kappa_r", "theta_r", "sigma_r", "rho_r"]

# ----------------------------
# Reference table: AE @ M=1e5
# ----------------------------
ref = df[(df["method"] == "Almost Exact") & (df["M"] == M_REF)][KEYS + ["price"]].copy()
ref = ref.rename(columns={"price": "price_ref"})

if ref.empty:
    raise RuntimeError(f"No reference rows found for Almost Exact with M={M_REF} in the CSV.")

# ----------------------------
# Merge reference into Euler/AE rows
# ----------------------------
work = df[df["method"].isin(["Euler", "Almost Exact"])].copy()
work = work.merge(ref, on=KEYS, how="left")

# Keep only rows that have a matching reference
work = work.dropna(subset=["price_ref"])

# Absolute error vs reference
work["abs_err"] = (work["price"] - work["price_ref"]).abs()

# ----------------------------
# 1) Error vs dt curve (mean over all parameter sets and rho)
# ----------------------------
# Group by (method, dt)
work["dt_r"] = work["dt"].round(12)

agg = (work.groupby(["method", "dt_r"], as_index=False)
          .agg(mean_abs_err=("abs_err", "mean"),
               std_abs_err=("abs_err", "std"),
               n=("abs_err", "size"))
          .sort_values(["method", "dt_r"]))

# Make dt order from large -> small so it reads naturally (coarse -> fine)
dt_order = sorted(agg["dt_r"].unique(), reverse=True)

plt.figure(figsize=(9, 5.5))
for method, marker in [("Euler", "o"), ("Almost Exact", "s")]:
    sub = agg[agg["method"] == method].set_index("dt_r").reindex(dt_order).reset_index()
    plt.plot(sub["dt_r"], sub["mean_abs_err"], marker=marker, label=method)

plt.xlabel("dt")
plt.ylabel("Mean absolute error vs AE(M=1e5)")
plt.title("Error vs dt (reference: Almost Exact, M=1e5)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(OUT_ERR_PDF)
plt.show()

print("Saved:", OUT_ERR_PDF)
print(agg)

# ----------------------------
# 2) Histogram: Euler deltaP vs reference (signed)
# ----------------------------
hist = work[(work["method"] == "Euler") & (work["M"] == HIST_M)].copy()
if hist.empty:
    raise RuntimeError(f"No Euler rows found with M={HIST_M}. Change HIST_M or check CSV.")

hist["deltaP"] = hist["price"] - hist["price_ref"]   # signed difference

plt.figure(figsize=(9, 5.5))
plt.hist(hist["deltaP"], bins=40)
plt.xlabel(f"ΔP = P_Euler(M={HIST_M}) − P_AE(M=1e5)")
plt.ylabel("Count")
plt.title(f"Distribution of ΔP over parameter sets (and rho)")
plt.grid(True)
plt.tight_layout()
plt.savefig(OUT_HIST_PDF)
plt.show()

print("Saved:", OUT_HIST_PDF)
print(hist[["kappa","theta","sigma","rho","M","dt","price","price_ref","deltaP"]].head())