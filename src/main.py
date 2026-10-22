# run: python -m src.main

import random
from pathlib import Path

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
DEFAULT_SEED = 42


def run_ga(start, ship, vehicles, rng, show, **params):
    return ga.search(start, ship, vehicles, rng, on_iteration=show, **params)


def run_sa(start, ship, vehicles, rng, show, **params):
    result = sa.search(start, ship, vehicles, rng, on_iteration=show, **params)
    print(f"Stuck       : {result.stuck_count} kali")
    return result


def run_hc_steepest(start, ship, vehicles, rng, show):
    return hc_steepest.search(start, ship, vehicles, vehicles, on_iteration=show)


def run_hc_random_restart(start, ship, vehicles, rng, show, **params):
    return hc_random_restart.search(
        start, ship, vehicles, rng, on_iteration=show, **params
    )


def run_hc_sideways(start, ship, vehicles, rng, show, **params):
    return hc_sideways.search(start, ship, vehicles, on_iteration=show, **params)


def run_hc_stochastic(start, ship, vehicles, rng, show):
    return hc_stochastic.search(start, ship, vehicles, rng, on_iteration=show)


# dict choice tui
ALGORITHMS = {
    "2": ("Genetic Algorithm", run_ga),
    "3": ("Simulated Annealing", run_sa),
    "4": ("HC Steepest Ascent", run_hc_steepest),
    "5": ("HC Random Restart", run_hc_random_restart),
    "6": ("HC Sideways Move", run_hc_sideways),
    "7": ("HC Stochastic", run_hc_stochastic),
}


def prompt_number(label, default, valid):
    while True:
        text = input(f"{label} [{default}]: ").strip()
        try:
            value = int(text) if text else default
            if valid(value):
                return value
        except ValueError:
            pass
        print("Nilai tidak valid. Ikuti batas pada prompt.")


def prompt_parameters(algorithm):
    print("Tekan Enter untuk nilai default.")
    if algorithm is run_ga:
        return {
            "max_iter": prompt_number("max_iter (>= 0)", 300, lambda n: n >= 0),
            "population_size": prompt_number(
                "population_size (>= 1)", 50, lambda n: n >= 1
            ),
        }
    if algorithm is run_sa:
        max_iter = prompt_number("max_iter (>= 0)", 20000, lambda n: n >= 0)
        print("SA dapat berhenti sebelum max_iter jika temperatur mencapai t_min.")
        return {"max_iter": max_iter}
    if algorithm is run_hc_random_restart:
        return {
            "max_restart": prompt_number(
                "max_restart (>= 1, termasuk run awal)", 10, lambda n: n >= 1
            ),
        }
    if algorithm is run_hc_sideways:
        return {
            "max_sideways_moves": prompt_number(
                "max_sideways_moves (>= 0)", 10, lambda n: n >= 0
            ),
        }
    print("Tidak ada parameter tambahan untuk varian ini.")
    return {}


# display per iterasi: n, current value, feasibility, kendaraan inside.
def show_iteration(ship, vehicles):
    def show(iteration, state, value):
        inside = [vid for vid, p in state.placements.items() if p.is_inside()]
        feasible = "feasible" if is_feasible(state, ship, vehicles) else "infeasible"
        print(
            f"Iterasi {iteration:>4} | value {value:>8} | {feasible:<10} | inside: {', '.join(inside)}"
        )

    return show


def run(name, algorithm, ship, vehicles, seed, **params):
    print(f"\n--- {name} ---")
    if params:
        print(f"Parameter   : {params}")

    visualize = input("Simpan visualisasi? [y/N]: ").strip().lower() == "y"
    rng = random.Random(seed)
    start = random_initial_state(ship, vehicles, rng)
    result = algorithm(
        start, ship, vehicles, rng, show_iteration(ship, vehicles), **params
    )

    print(f"Value awal  : {objective(result.initial_state, ship, vehicles)}")
    print(f"Value akhir : {result.final_value}")
    print(f"Feasible    : {is_feasible(result.final_state, ship, vehicles)}")
    print(f"Iterasi     : {result.iterations}")
    if "population_size" in getattr(result, "extra", {}):  # khusus GA, yg punya cuma GA
        print(f"Populasi    : {result.extra['population_size']}")
    print(f"Durasi      : {result.duration_s:.3f} s")

    if visualize:
        from src.visualize import draw_run  # only import matplotlib kalo perlu

        out_dir = Path("output") / name.lower().replace(" ", "_")
        out_dir.mkdir(parents=True, exist_ok=True)
        suffix = "".join(f"{k.removesuffix('_size')}{v}" for k, v in params.items())
        path = out_dir / f"seed{seed}{suffix}.png"
        label = "_".join(f"{k}={v}" for k, v in params.items()) or "default"
        draw_run(name, label, seed, result, ship, vehicles, path)
        print(f"Visualisasi : {path}")


def main():
    ship, vehicles = load_problem(DATA_PATH)
    while True:
        print("\n== Penempatan Kendaraan ==")
        print("1. Semua algoritma")
        for key, (name, _) in ALGORITHMS.items():
            print(f"{key}. {name}")
        print("0. Keluar")

        choice = input("Pilih: ").strip()
        if choice == "0":
            return
        if choice != "1" and choice not in ALGORITHMS:
            print("Pilihan tidak valid.")
            continue
        seed = prompt_number("Seed (integer)", DEFAULT_SEED, lambda n: True)

        selected = ALGORITHMS.values() if choice == "1" else [ALGORITHMS[choice]]
        for name, algorithm in selected:
            print(f"\nParameter {name}")
            params = prompt_parameters(algorithm)
            run(name, algorithm, ship, vehicles, seed, **params)


if __name__ == "__main__":
    main()
