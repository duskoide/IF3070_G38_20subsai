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
    max_sideways_moves: int = 10,
) -> SearchResult:
    """Batas sideways dihitung berturut-turut dan direset setelah uphill."""
    if max_sideways_moves < 0:
        raise ValueError("max_sideways_moves tidak boleh negatif")

    start_time = time.perf_counter()
    current = initial_state
    current_value = objective(current, ship, vehicles_by_id)
    history = [current_value]
    iterations = 0
    sideways_moves = 0
    stop_reason = "local_optimum"
    best_state: State | None = None
    best_value: float | None = None
    if is_feasible(current, ship, vehicles_by_id):
        best_state, best_value = current, current_value

    while True:
        best_neighbor = None
        best_neighbor_value = current_value

        for neighbor in all_neighbors(current, ship, vehicles_by_id):
            if neighbor == current:
                continue
            neighbor_value = objective(neighbor, ship, vehicles_by_id)

            if neighbor_value > best_neighbor_value or (
                best_neighbor is None and neighbor_value == current_value
            ):
                best_neighbor = neighbor
                best_neighbor_value = neighbor_value

            if (best_value is None or neighbor_value > best_value) and is_feasible(
                neighbor, ship, vehicles_by_id
            ):
                best_state, best_value = neighbor, neighbor_value

        if best_neighbor is None:  # local optimum
            break

        if best_neighbor_value == current_value:
            if sideways_moves >= max_sideways_moves:
                stop_reason = "max_sideways_moves"
                break
            sideways_moves += 1
        else:
            sideways_moves = 0

        current, current_value = best_neighbor, best_neighbor_value
        iterations += 1
        history.append(current_value)

        if sideways_moves > 0 and sideways_moves >= max_sideways_moves:
            stop_reason = "max_sideways_moves"
            break

    duration_s = time.perf_counter() - start_time

    return SearchResult(
        initial_state=initial_state,
        final_state=best_state,
        final_value=best_value,
        history=history,
        iterations=iterations,
        duration_s=duration_s,
        extra={
            "max_sideways_moves": max_sideways_moves,
            "stop_reason": stop_reason,
        },
    )