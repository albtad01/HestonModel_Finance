import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

# =========================================================
# CONFIG
# =========================================================
CSV_PATH     = "results/paths.csv"          # prima colonna = t, resto = paths
OUTPUT_MP4   = "results/heston_paths_trading_style.mp4"
MAX_PATHS    = 20                   # quanti path mostrare
STEP         = 10                   # prendo un frame ogni STEP time-step
FPS          = 15                   # fotogrammi al secondo (più basso = più lento)
INTERVAL_MS  = 120                  # intervallo tra frame per preview

# =========================================================
# LOAD DATA
# =========================================================
df = pd.read_csv(CSV_PATH)
time = df.iloc[:, 0].values
paths_all = df.iloc[:, 1:].values

paths = paths_all[:, :MAX_PATHS]
n_steps, n_paths = paths.shape
print(f"Loaded {n_paths} paths with {n_steps} time steps")

frame_indices = np.arange(0, n_steps, STEP)
if frame_indices[-1] != n_steps - 1:
    frame_indices = np.append(frame_indices, n_steps - 1)

K = 1.0  # Strike

# =========================================================
# PLOT STYLE (dark trading style)
# =========================================================
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.edgecolor": "#374151",
    "axes.labelcolor": "#e5e7eb",
    "xtick.color": "#9ca3af",
    "ytick.color": "#9ca3af",
    "figure.facecolor": "#020617",
    "axes.facecolor": "#020617",
    "grid.color": "#1f2933",
    "grid.linestyle": "--",
    "grid.alpha": 0.35,
    "legend.edgecolor": "#4b5563",
})

fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
fig.patch.set_facecolor("#020617")
ax.set_facecolor("#020617")

ax.grid(True)

ax.set_xlim(time[0], time[-1])
y_min = np.min(paths)
y_max = np.max(paths)
margin = 0.05 * (y_max - y_min)
ax.set_ylim(y_min - margin, y_max + margin)

ax.set_xlabel("t", fontsize=11)
ax.set_ylabel("S(t)", fontsize=11)

# 👉 QUI CAMBI IL TITOLO
# ad es.: "Heston Model – Exact Monte Carlo Simulation"
ax.set_title("Heston Model – Exact Monte Carlo Simulation", fontsize=13, color="#e5e7eb")

# colori
path_color   = "#4b5563"   # grigio scuro per i path
mean_color   = "#22d3ee"   # azzurro per la media
strike_color = "#ffffff"   # BIANCO per lo strike

# linee dei path
lines = []
for _ in range(n_paths):
    (line,) = ax.plot([], [], lw=1.0, alpha=0.45,
                      color=path_color, zorder=2)
    lines.append(line)

# linea della media
(mean_line,) = ax.plot([], [], lw=2.0, color=mean_color,
                       label="Mean S(t)", zorder=4)

# linea dello strike (bianca tratteggiata)
strike_line = ax.axhline(K, color=strike_color, lw=1.2,
                         ls="--", label="Strike K", zorder=3)

# testo del tempo + media corrente
time_text = ax.text(
    0.98, 0.06, "",
    transform=ax.transAxes,
    ha="right", va="bottom",
    fontsize=9, color="#9ca3af"
)

# formule del modello
eq_text = ax.text(
    0.02, 0.96,
    r"$S_{t+\Delta t} = S_t + r S_t \Delta t + \sqrt{v_t}\,S_t \sqrt{\Delta t}\,"
    r"(\rho G_1 + \sqrt{1-\rho^2}\,G_2)$"
    "\n"
    r"$v_{t+\Delta t} = \frac{\sigma^2 (1-e^{-\kappa \Delta t})}{2\kappa}\,"
    r"\mathcal{G}(d+N)$",
    transform=ax.transAxes,
    fontsize=7.5,
    color="#9ca3af",
    ha="left", va="top"
)

# legenda
legend = ax.legend(
    loc="upper right",
    frameon=True,
    fontsize=9,
    facecolor="#020617",
    labelcolor="#e5e7eb"
)
legend.get_frame().set_alpha(0.9)

# =========================================================
# INIT & UPDATE
# =========================================================
def init():
    for line in lines:
        line.set_data([], [])
    mean_line.set_data([], [])
    time_text.set_text("")
    return lines + [mean_line, strike_line, time_text, eq_text]

def update(frame_idx):
    k = frame_indices[frame_idx]
    t_slice = time[:k + 1]

    # aggiorna tutti i path
    for i, line in enumerate(lines):
        line.set_data(t_slice, paths[:k + 1, i])

    # aggiorna la media
    mean_path = paths[:k + 1, :].mean(axis=1)
    current_mean = mean_path[-1]
    mean_line.set_data(t_slice, mean_path)

    # testo: tempo + E[S(t)]
    time_text.set_text(f"t = {time[k]:.3f} \n   E[S(t)] = {current_mean:.3f}")

    return lines + [mean_line, strike_line, time_text, eq_text]

# =========================================================
# ANIMATION
# =========================================================
anim = FuncAnimation(
    fig,
    update,
    frames=len(frame_indices),
    init_func=init,
    blit=True,
    interval=INTERVAL_MS,
)

writer = FFMpegWriter(fps=FPS, bitrate=2200)
print(f"Saving to {OUTPUT_MP4} ...")
anim.save(OUTPUT_MP4, writer=writer)
print("Done.")

plt.close(fig)