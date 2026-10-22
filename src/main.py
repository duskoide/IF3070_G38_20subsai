# run: python -m src.main

import random

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

def run_ga(start, ship, vehicles, rng, show):
    return ga.search(start, ship, vehicles, rng, on_iteration=show)


def run_sa(start, ship, vehicles, rng, show):
    return sa.search(start, ship, vehicles, rng, on_iteration=show)


def run_hc_steepest(start, ship, vehicles, rng, show):
    return hc_steepest.search(start, ship, vehicles, vehicles, on_iteration=show)


def run_hc_random_restart(start, ship, vehicles, rng, show):
    return hc_random_restart.search(start, ship, vehicles, rng, on_iteration=show)


def run_hc_sideways(start, ship, vehicles, rng, show):
    return hc_sideways.search(start, ship, vehicles, on_iteration=show)


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


# display per iterasi: n, current value, feasibility, kendaraan inside.
def show_iteration(ship, vehicles):
    def show(iteration, state, value):
        inside = [vid for vid, p in state.placements.items() if p.is_inside()]
        feasible = "feasible" if is_feasible(state, ship, vehicles) else "infeasible"
        print(f"Iterasi {iteration:>4} | value {value:>8} | {feasible:<10} | inside: {', '.join(inside)}")
    return show


def run(name, algorithm, ship, vehicles, seed):
    print(f"\n--- {name} ---")
    rng = random.Random(seed)
    start = random_initial_state(ship, vehicles, rng)
    result = algorithm(start, ship, vehicles, rng, show_iteration(ship, vehicles))

    print(f"Value awal  : {objective(start, ship, vehicles)}")
    print(f"Value akhir : {result.final_value}")
    print(f"Feasible    : {is_feasible(result.final_state, ship, vehicles)}")
    print(f"Iterasi     : {result.iterations}")
    if hasattr(result, "stuck_count"):  #cmn buat SA
        print(f"Stuck       : {result.stuck_count} kali")
    print(f"Durasi      : {result.duration_s:.3f} s")


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
        seed_text = input(f"Seed [{DEFAULT_SEED}]: ").strip()
        seed = int(seed_text) if seed_text else DEFAULT_SEED

        selected = ALGORITHMS.values() if choice == "1" else [ALGORITHMS[choice]]
        for name, algorithm in selected:
            run(name, algorithm, ship, vehicles, seed)


if __name__ == "__main__":
    main()
