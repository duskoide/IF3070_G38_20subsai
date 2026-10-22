import math
import random
import time
from dataclasses import dataclass
from src.model import Orientation, Placement, Ship, State, Vehicle
from src.neighbors import random_neighbor
from src.objective import is_feasible, objective

MOVE_WEIGHTS = {"swap": 1, "move": 1, "move_out": 0.1, "rotate": 1}

@dataclass
class SearchResult:
    initial_state: State
    final_state: State
    final_value: float
    last_state: State
    history: list[float]
    probabilities: list[float]
    temperatures: list[float]
    iterations: int
    stuck_count: int
    duration_s: float
    params: dict
    extra: dict

def empty_state(vehicles_by_id: dict[str, Vehicle]) -> State:
    placements = {}
    for vehicle_id in vehicles_by_id:
        placements[vehicle_id] = Placement(
            vehicle_id=vehicle_id,
            x=None,
            y=None,
            orientation=Orientation.HORIZONTAL,
        )
    return State(placements)

def search(
    initial_state: State,
    ship: Ship,
    vehicles_by_id: dict[str, Vehicle],
    rng: random.Random | None = None,
    *,
    on_iteration=None,
    t0: float = 1000.0,
    alpha: float = 0.999,
    t_min: float = 0.01,
    max_iter: int = 20000,
    stuck_k: int = 500,
    seed: int | None = None,
    move_weights: dict[str, float] | None = None,
) -> SearchResult:
    if move_weights is None:
        move_weights = MOVE_WEIGHTS

    #rng dari main.py dipake langsung, kalo nggak ada, dibuat dari seed
    if rng is None:
        rng = random.Random(seed)
    start_time = time.perf_counter()

    current_state = initial_state
    current_value = objective(current_state, ship, vehicles_by_id)
    best_value = current_value

    if is_feasible(current_state, ship, vehicles_by_id):
        best_feasible_state = current_state
        best_feasible_value = current_value
    else:
        best_feasible_state = None
        best_feasible_value = -math.inf

    temperature = t0
    value_history = [current_value]   
    probability_history = []          
    temperature_history = []          
    stuck_count = 0
    iterations_without_improvement = 0
    iteration = 0

    if on_iteration is not None:
        on_iteration(0, current_state, current_value)

    while iteration < max_iter and temperature > t_min:
        neighbor = random_neighbor(current_state, ship, vehicles_by_id, rng, move_weights)
        if neighbor is None:
            break
        iteration = iteration + 1

        neighbor_value = objective(neighbor, ship, vehicles_by_id)
        delta = neighbor_value - current_value

        if delta >= 0:
            probability = 1.0
        else:
            probability = math.exp(delta / temperature)

        random_number = rng.random()
        if random_number <= probability:
            current_state = neighbor
            current_value = neighbor_value
        if current_value > best_value:
            best_value = current_value
            iterations_without_improvement = 0
        else:
            iterations_without_improvement = iterations_without_improvement + 1
            if iterations_without_improvement == stuck_k:
                stuck_count = stuck_count + 1
                iterations_without_improvement = 0

        if current_value > best_feasible_value:
            if is_feasible(current_state, ship, vehicles_by_id):
                best_feasible_state = current_state
                best_feasible_value = current_value

        value_history.append(current_value)
        probability_history.append(probability)
        temperature_history.append(temperature)

        if on_iteration is not None:
            on_iteration(iteration, current_state, current_value)

        temperature = temperature * alpha

    if best_feasible_state is None:
        best_feasible_state = empty_state(vehicles_by_id)
        best_feasible_value = objective(best_feasible_state, ship, vehicles_by_id)

    duration = time.perf_counter() - start_time

    if temperature <= t_min:
        stop_reason = "t_min"
    elif iteration >= max_iter:
        stop_reason = "max_iter"
    else:
        stop_reason = "no_neighbor"

    # ringkasan untuk visualize.py, bentuknya sama dengan extra di HC/GA
    extra = {
        "stop_reason": stop_reason,
        "stuck_count": stuck_count,
        "stuck_k": stuck_k,
        "t0": t0,
        "alpha": alpha,
        "t_min": t_min,
        "max_iter": max_iter,
    }

    params = {
        "t0": t0,
        "alpha": alpha,
        "t_min": t_min,
        "max_iter": max_iter,
        "stuck_k": stuck_k,
        "seed": seed,
        "move_weights": dict(move_weights),
    }

    result = SearchResult(
        initial_state=initial_state,
        final_state=best_feasible_state,
        final_value=best_feasible_value,
        last_state=current_state,
        history=value_history,
        probabilities=probability_history,
        temperatures=temperature_history,
        iterations=iteration,
        stuck_count=stuck_count,
        duration_s=duration,
        params=params,
        extra=extra,
    )
    return result