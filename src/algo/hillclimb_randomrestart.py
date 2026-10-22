import random
from time import perf_counter
from dataclasses import dataclass

from src.neighbors import all_neighbors
from src.initial import random_initial_state
from src.objective import is_feasible, objective
from src.model import State, Vehicle, Ship
from src.algo.hillclimb_steepestascent import search as steepest_ascent

@dataclass
class SearchResult:
    initial_state: State
    final_state: State | None
    final_value: float | None
    history: list[float]
    iterations: int
    duration_s: float
    extra: dict

def search(
    initial_state: State,
    ship: Ship,
    vehicles: dict[str, Vehicle],
    rng: random.Random,
    max_restart: int = 10,
    *,
    on_iteration=None,
) -> SearchResult:
    """Callback memakai jumlah move kumulatif, termasuk state awal tiap restart."""

    if max_restart < 1:
        raise ValueError("max_restart minimal harus 1 yak")

    start_time = perf_counter()
    vehicles_by_id = vehicles
    best_state: State | None = None
    best_value: float | None = None
    history: list[float] = []
    restart_starts: list[int] = [] # mencatat posisi awal tiap restart (penanda mulai restart baru)
    iterations_per_restart: list[int] = []
    restart_final_values: list[float | None] = []  # value akhir tiap restart
    total_iterations = 0

    def report_iteration(iteration, state, value):
        if on_iteration is not None:
            on_iteration(total_iterations + iteration, state, value)

    for restart in range(max_restart):
        start = (
            initial_state
            if restart == 0
            else random_initial_state(ship, vehicles, rng)
        )

        run = steepest_ascent(
            start, ship, vehicles, vehicles_by_id,
            on_iteration=report_iteration if on_iteration is not None else None,
        )

        # mencatat seluruh history dari setiap restart
        restart_starts.append(len(history)) # mencatat panjang history sebelum restart baru dimulai
        history.extend(run.history)
        iterations_per_restart.append(run.iterations)
        restart_final_values.append(run.final_value)
        total_iterations += run.iterations


        if (
            run.final_state is not None
            and run.final_value is not None
            and (best_value is None or run.final_value > best_value)
        ):
            best_state, best_value = run.final_state, run.final_value

    return SearchResult(
        initial_state=initial_state,
        final_state=best_state,
        final_value=best_value,
        history=history,
        iterations=total_iterations,
        duration_s=perf_counter() - start_time,
        extra={
            "variant": "random_restart",
            "max_restart": max_restart,
            "restart_count": max_restart,
            "iterations_per_restart": iterations_per_restart,
            "restart_starts": restart_starts,
            "restart_final_values": restart_final_values,
        },
    )
