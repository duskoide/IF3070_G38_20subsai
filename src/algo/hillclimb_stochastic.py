import random
import time
from dataclasses import dataclass

from src.model import Ship, State, Vehicle
from src.neighbors import all_neighbors
from src.objective import is_feasible, objective


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
    vehicles_by_id: dict[str, Vehicle],
    rng: random.Random, #dipakai untuk memilih 1 tetangga uphill
    *,
    on_iteration=None,
) -> SearchResult:
    t = time.perf_counter() #untuk mengukur selang waktu >> dipake hitung durasi
    current = initial_state
    current_value = objective(current, ship, vehicles_by_id)
    best_state: State | None = None
    best_value: float | None = None

    if is_feasible(current, ship, vehicles_by_id):
        best_state = current
        best_value = current_value

    history = [current_value]
    iterations = 0
    if on_iteration is not None:
        on_iteration(0, current, current_value)
    while True:
        uphill = []
        for n in all_neighbors(current, ship, vehicles_by_id):
            val = objective(n, ship, vehicles_by_id)
            if val > current_value:
                uphill.append((n, val))

            # Simpan neighbor feasible 
            if (best_value is None or val > best_value) and is_feasible(
                n, ship, vehicles_by_id
            ):
                best_state, best_value = n, val

        if not uphill:
            break

        current, current_value = rng.choice(uphill)
        iterations += 1
        history.append(current_value)
        if on_iteration is not None:
            on_iteration(iterations, current, current_value)


    return SearchResult(
        initial_state=initial_state,
        final_state=best_state,
        final_value=best_value,
        history=history,
        iterations=iterations, 
        duration_s=time.perf_counter() - t,
        extra={"stop_reason": "local_optimum"},
    )