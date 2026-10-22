from time import perf_counter
from dataclasses import dataclass

from src.neighbors import all_neighbors
from src.objective import is_feasible, objective
from src.model import State, Vehicle, Ship

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
    start: State,
    ship: Ship,
    vehicles: dict[str, Vehicle],
    vehicles_by_id: dict[str, Vehicle],
    on_iteration=None,
) -> SearchResult:
    start_time = perf_counter()
    current = start
    current_value = objective(current, ship, vehicles_by_id)
    history = [current_value]
    iterations = 0
    if on_iteration is not None:
        on_iteration(0, current, current_value)

    best_state: State | None = None
    best_value: float | None = None
    if is_feasible(current, ship, vehicles_by_id):
        best_state, best_value = current, current_value

    while True:
        best_neighbor = None
        best_neighbor_value = current_value

        for neighbor in all_neighbors(current, ship, vehicles):
            neighbor_value = objective(neighbor, ship, vehicles_by_id)

            if neighbor_value > best_neighbor_value:
                best_neighbor = neighbor
                best_neighbor_value = neighbor_value

        if best_neighbor is None:  # local optimum
            break

        current, current_value = best_neighbor, best_neighbor_value
        iterations += 1
        history.append(current_value)
        if on_iteration is not None:
            on_iteration(iterations, current, current_value)

        # catat best feasible dari state yang dikunjungi
        if (best_value is None or current_value > best_value) and is_feasible(
            current, ship, vehicles_by_id
        ):
            best_state, best_value = current, current_value

    return SearchResult(
        initial_state=start,
        final_state=best_state,
        final_value=best_value,
        history=history,
        iterations=iterations,
        duration_s=perf_counter() - start_time,
        extra={}
    )
