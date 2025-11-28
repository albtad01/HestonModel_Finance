#!/usr/bin/env python3
# analyze_benchmark.py
#
# Analisi dei risultati di heston_3.cu:
# - boxplot dei tempi per metodo e M
# - scatter time_ms vs (kappa, theta, sigma)
# - confronto prezzi Euler(M=1000) vs AlmostExact(M=30)

import sys
import pandas as pd
import matplotlib.pyplot as plt

def main(csv_path: str = "benchmark_results.csv"):
    # ------------------------------------------------------------------
    # 1. Caricamento dati
    # ------------------------------------------------------------------
    df = pd.read_csv(csv_path)

    # Controllino rapido
    print("Prime righe del dataset:")
    print(df.head(), "\n")

    # ------------------------------------------------------------------
    # 2. Boxplot time_ms per metodo e M
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    for ax, M in zip(axes, sorted(df["M"].unique())):
        sub = df[df["M"] == M]
        data = [
            sub[sub["method"] == "Euler"]["time_ms"],
            sub[sub["method"] == "Almost Exact"]["time_ms"],
        ]
        ax.boxplot(data, labels=["Euler", "Almost Exact"])
        ax.set_title(f"M = {M}")
        ax.set_ylabel("time_ms")
        ax.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle("Execution time by method and M")
    fig.tight_layout()
    fig.savefig("boxplot_time_by_method_M.png", dpi=200)

    # ------------------------------------------------------------------
    # 3. Scatter time_ms vs kappa, theta, sigma
    #    (coloriamo per metodo+M)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    params = ["kappa", "theta", "sigma"]
    titles = [r"$\kappa$", r"$\theta$", r"$\sigma$"]

    # Serie separate per metodo e M
    for (method, M), sub in df.groupby(["method", "M"]):
        label = f"{method}, M={M}"
        for ax, p, title in zip(axes, params, titles):
            ax.scatter(sub[p], sub["time_ms"], alpha=0.6, label=label)

    for ax, p, title in zip(axes, params, titles):
        ax.set_xlabel(title)
        ax.set_ylabel("time_ms")
        ax.grid(True, linestyle="--", alpha=0.4)

    # Legend solo sul primo asse per non impazzire
    axes[0].legend(fontsize=8)
    fig.suptitle("Execution time vs parameters")
    fig.tight_layout()
    fig.savefig("scatter_time_vs_params.png", dpi=200)

    # ------------------------------------------------------------------
    # 4. Confronto prezzi:
    #    Euler(M=1000) vs AlmostExact(M=30)
    # ------------------------------------------------------------------
    euler_1000 = df[(df["method"] == "Euler") & (df["M"] == 1000)].copy()
    ae_30      = df[(df["method"] == "Almost Exact") & (df["M"] == 30)].copy()

    # Merge su (kappa, theta, sigma, rho)
    merge_cols = ["kappa", "theta", "sigma", "rho"]
    merged = pd.merge(
        euler_1000,
        ae_30,
        on=merge_cols,
        suffixes=("_eu1000", "_ae30"),
    )

    # (Opzionale) filtro parametri "realistici" – puoi cambiare le soglie
    realistic_mask = (
        (merged["kappa"] >= 0.5) & (merged["kappa"] <= 5.0)
        & (merged["theta"] >= 0.05) & (merged["theta"] <= 0.3)
        & (merged["sigma"] >= 0.1) & (merged["sigma"] <= 0.7)
    )
    merged_real = merged[realistic_mask].copy()

    print(f"\nNumero combinazioni dopo filtro 'realistico': {len(merged_real)}")

    # Differenza e errore relativo
    merged_real["price_diff"] = (
        merged_real["price_ae30"] - merged_real["price_eu1000"]
    )
    merged_real["rel_error"] = (
        merged_real["price_diff"] / merged_real["price_eu1000"]
    )

    print(
        "\nStatistiche price_diff (AE(M=30) - Euler(M=1000)):\n",
        merged_real["price_diff"].describe(),
    )
    print(
        "\nStatistiche errore relativo (rispetto a Euler M=1000):\n",
        merged_real["rel_error"].describe(),
    )

    # Istogramma differenze di prezzo
    plt.figure(figsize=(6, 4))
    plt.hist(merged_real["price_diff"], bins=20, edgecolor="black", alpha=0.7)
    plt.xlabel("price_AE(M=30) - price_Euler(M=1000)")
    plt.ylabel("Frequency")
    plt.title("Histogram of price differences")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig("hist_price_diff.png", dpi=200)

    # Istogramma errore relativo
    plt.figure(figsize=(6, 4))
    plt.hist(merged_real["rel_error"], bins=20, edgecolor="black", alpha=0.7)
    plt.xlabel("Relative error (AE30 vs Euler1000)")
    plt.ylabel("Frequency")
    plt.title("Histogram of relative errors")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig("hist_rel_error.png", dpi=200)

    print("\nFigure salvate come:")
    print("  - boxplot_time_by_method_M.png")
    print("  - scatter_time_vs_params.png")
    print("  - hist_price_diff.png")
    print("  - hist_rel_error.png")

    # Mostra tutto a schermo (se lanci da notebook / locale)
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main()
