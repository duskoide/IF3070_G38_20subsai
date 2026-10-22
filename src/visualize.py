# run: .venv/bin/python -B -m src.visualize [nama_algoritma ...] or uv run -m src.visualize [nama_algoritma ...]
# output: output/<nama_algoritma>/<konfigurasi>_seed<seed>.png

import random
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # simpan ke file, tanpa jendela
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from src.algo import genetic_algorithm as ga
from src.algo import hillclimb_randomrestart as hc_random_restart
from src.algo import hillclimb_sideways as hc_sideways
from src.algo import hillclimb_steepestascent as hc_steepest
from src.algo import hillclimb_stochastic as hc_stochastic
from src.algo import simulated_annealing as sa
from src.initial import random_initial_state
from src.loader import load_problem
from src.objective import is_feasible, objective

DATA_PATH = "src/data/sample.json"
SEEDS = [67, 69, 420]  # 3 run per konfigurasi
SEEDS_BY_ALGORITHM = {"sa": [1, 2, 3]}  # seed khusus per algoritma (SA mengikuti laporan SA)

# GA eksperimen A: populasi 50, iterasi divariasikan; B: iterasi 300, populasi divariasikan
GA_CONFIGS = [(50, 100), (50, 300), (50, 1000), (20, 300), (100, 300)]

# nama -> list (label konfigurasi, fungsi run)
ALGORITHMS = {
    "hc_steepest": [("default", lambda s, ship, v, rng: hc_steepest.search(s, ship, v, v))],
    "hc_sideways": [("max_sideways=10", lambda s, ship, v, rng: hc_sideways.search(s, ship, v, 10))],
    "hc_random_restart": [("max_restart=10", lambda s, ship, v, rng: hc_random_restart.search(s, ship, v, rng, 10))],
    "hc_stochastic": [("default", lambda s, ship, v, rng: hc_stochastic.search(s, ship, v, rng))],
    "ga": [
        (f"pop={p}_iter={i}",
         lambda s, ship, v, rng, p=p, i=i: ga.search(s, ship, v, rng, population_size=p, max_iter=i))
        for p, i in GA_CONFIGS
    ],
    "sa": [("default", lambda s, ship, v, rng: sa.search(s, ship, v, rng))],
}


# grid geladak: x ke kanan (lebar), y ke bawah (panjang); kendaraan = kotak bergaris tepi
def draw_state(ax, title, state, ship, vehicles):
    ax.set_xlim(0, ship.w)
    ax.set_ylim(ship.l, 0)
    ax.set_aspect("equal")
    ax.set_xticks(range(ship.w + 1))
    ax.set_yticks(range(ship.l + 1))
    ax.tick_params(labelsize=6)
    ax.grid(color="#ddd", linewidth=0.5)
    ax.set_axisbelow(True)
    if state is None:
        ax.set_title(f"{title}\ntidak ada state feasible", fontsize=9)
        return
    for vid, p in state.placements.items():
        if p.is_inside():
            vw, vl = vehicles[vid].size_taken(p.orientation)
            ax.add_patch(Rectangle((p.x, p.y), vw, vl, fill=False, linewidth=1.2))
            ax.text(p.x + vw / 2, p.y + vl / 2, vid, ha="center", va="center",
                    fontsize=6, rotation=90 if vl > vw else 0)
    outside = [vid for vid, p in state.placements.items() if not p.is_inside()]
    ax.set_title(f"{title}\nvalue {objective(state, ship, vehicles)}, "
                 f"feasible {is_feasible(state, ship, vehicles)}", fontsize=9)
    ax.set_xlabel(textwrap.fill("Di luar kapal: " + ", ".join(outside), 45), fontsize=6)


def draw_run(name, label, seed, result, ship, vehicles, path):
    # SA punya plot tambahan e^(ΔE/T) di bawah plot objective (R-EXP-06)
    has_probability = hasattr(result, "probabilities")
    fig = plt.figure(figsize=(15, 7))
    grid = fig.add_gridspec(2 if has_probability else 1, 3, width_ratios=[2, 1, 1])
    ax_plot = fig.add_subplot(grid[0, 0])
    ax_start = fig.add_subplot(grid[:, 1])
    ax_final = fig.add_subplot(grid[:, 2])

    is_ga = "history_avg" in result.extra
    ax_plot.plot(result.history, label="maksimum populasi" if is_ga else "value")
    if is_ga:
        ax_plot.plot(result.extra["history_avg"], label="rata-rata populasi")
    ax_plot.set_xlabel("iterasi")
    ax_plot.set_ylabel("objective")
    ax_plot.legend()

    if has_probability:
        ax_prob = fig.add_subplot(grid[1, 0])
        iterations = range(1, len(result.probabilities) + 1)
        ax_prob.scatter(iterations, result.probabilities, s=1, label="min(1, e^(ΔE/T))")
        ax_prob.set_xlabel("iterasi")
        ax_prob.set_ylabel("e^(ΔE/T)")
        ax_prob.set_ylim(-0.05, 1.05)
        ax_prob.legend(loc="upper right")

    draw_state(ax_start, "State awal", result.initial_state, ship, vehicles)
    draw_state(ax_final, "State akhir", result.final_state, ship, vehicles)

    fig.suptitle(f"{name} | {label} | seed {seed} | value akhir {result.final_value} | "
                 f"iterasi {result.iterations} | durasi {result.duration_s:.3f} s")
    extra = {k: v for k, v in result.extra.items() if k != "history_avg"}
    ax_plot.set_title(textwrap.fill(str(extra), 100), fontsize=7, loc="left")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    ship, vehicles = load_problem(DATA_PATH)
    for name in sys.argv[1:] or ALGORITHMS:
        out_dir = Path("output") / name
        out_dir.mkdir(parents=True, exist_ok=True)
        for label, run in ALGORITHMS[name]:
            for seed in SEEDS_BY_ALGORITHM.get(name, SEEDS):
                rng = random.Random(seed)
                start = random_initial_state(ship, vehicles, rng)
                result = run(start, ship, vehicles, rng)
                path = out_dir / f"{label}_seed{seed}.png"
                draw_run(name, label, seed, result, ship, vehicles, path)
                print(f"{name} | {label} | seed {seed} | value {result.final_value} "
                      f"| {result.duration_s:.2f} s -> {path}")


if __name__ == "__main__":
    main()
