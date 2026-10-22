import dataclasses
import random
from time import perf_counter

from src.model import Orientation, Ship, State, Vehicle, Placement
from src.initial import random_initial_state
from src.neighbors import random_neighbor
from src.objective import is_feasible, objective


# Nilai awal rekomendasi
MUTATION_PROBABILITY = 0.1

# Pressure s rank roulette
RANK_PRESSURE = 1.5 # antara 1 sampe 2


@dataclasses.dataclass
class SearchResult:
    initial_state: State
    final_state: State
    final_value: float
    history: list[float]
    iterations: int
    duration_s: float
    extra: dict[str, object]


def initialize_population(
    initial_state: State | None,
    ship: Ship,
    vehicles_by_id: dict[str, Vehicle],
    rng: random.Random,
    population_size: int,
) -> list[State]:

    population = []
    if initial_state is None:
        for i in range (population_size):
            population.append(random_initial_state(ship, vehicles_by_id, rng))
    else:
        population.append(State(dict(initial_state.placements))) # new State dari initial_state
        for i in range (population_size - 1):
            population.append(random_initial_state(ship, vehicles_by_id, rng))
    return population

def evaluate_population(
    population: list[State],
    ship: Ship,
    vehicles_by_id: dict[str, Vehicle],
) -> list[float]:

    fitness = [ objective(s, ship, vehicles_by_id) for s in population]
    return fitness


def select_parent(
    population: list[State], fitness: list[float], rng: random.Random
) -> State:

    # Rank roulette
    n = len(population)
    if n == 0 or n != len(fitness):
        raise ValueError("populasi kosong atau panjang tidak sama dengan fitness")
    if n == 1:
        return population[0]
    s = RANK_PRESSURE
    order = sorted(range(n), key=lambda i: fitness[i])
    weights = [0.0] * n
    for r, i in enumerate(order): # r ini rank/position per i
        weights[i] = (2 - s) + 2 * (s - 1) * r / (n - 1)
    return rng.choices(population, weights=weights, k=1)[0]

def crossover(
    parent_a: State,
    parent_b: State,
    vehicles_by_id: dict[str, Vehicle],
    rng: random.Random,
) -> State:

    children_id = list(vehicles_by_id)
    n = len(children_id)
    if n >= 2:
        c = rng.randint(1, n-1)

        placements = {}
        for i, vehicle_id in enumerate(children_id):
            if i < c:
                parent = parent_a
            else:
                parent = parent_b
            placements[vehicle_id] = parent.placements[vehicle_id]

        return State(placements)
    else:
        return State(dict(parent_a.placements))


def mutate(
    child: State, ship: Ship, vehicles_by_id: dict[str, Vehicle], rng: random.Random
) -> State:

    potential_mutation = random_neighbor(child, ship, vehicles_by_id, rng)
    return potential_mutation if potential_mutation is not None else child


def update_best_feasible(
    population: list[State],
    fitness: list[float],
    ship: Ship,
    vehicles_by_id: dict[str, Vehicle],
    best_state: State,
    best_value: float,
) -> tuple[State, float]:

    current_best_state = best_state
    current_best_value = best_value

    for i in range (len(population)):
        if fitness[i] > current_best_value and is_feasible(population[i], ship, vehicles_by_id):
            current_best_state = State(dict(population[i].placements))
            current_best_value = fitness[i]

    return current_best_state, current_best_value


def search(
    initial_state: State | None,
    ship: Ship,
    vehicles_by_id: dict[str, Vehicle],
    rng: random.Random,
    *, # pisahin so that 3 args di bawah perlu explicit name di parameter, misal: search(..., seed = 67)
    population_size: int = 50,
    max_iter: int = 300,
    seed: int | None = None,
) -> SearchResult:
    if isinstance(population_size, bool) or not isinstance(population_size, int) or population_size < 1:
        raise ValueError("population_size harus integer >= 1")

    if isinstance(max_iter, bool) or not isinstance(max_iter, int) or max_iter < 0:
        raise ValueError("max_iter harus integer >= 0")

    for key, vehicle in vehicles_by_id.items():
        if key != vehicle.id:
            raise ValueError(f"Key {key!r} tidak sama dengan vehicle.id {vehicle.id!r}")

    start_time = perf_counter()

    # start new population states
    population = initialize_population(initial_state, ship, vehicles_by_id, rng, population_size)

    # list of weights for each state di population
    fitness = evaluate_population(population, ship, vehicles_by_id)

    current_idx = fitness.index(max(fitness)) # since fitness paralel dgn population, idxnya sama
    initial_result = State(dict(population[current_idx].placements)) # best current population
    initial_value = fitness[current_idx]

    best_state = State({vehicle_id: Placement(vehicle_id, None, None, Orientation.HORIZONTAL) for vehicle_id in vehicles_by_id})
    best_value = objective(best_state, ship, vehicles_by_id)
    best_state, best_value = update_best_feasible(population, fitness, ship, vehicles_by_id, best_state, best_value)

    history = [max(fitness)]
    history_avg = [sum(fitness) / len(fitness)]

    for i in range(max_iter):
        new_population = []
        while len(new_population) < population_size:
            a = select_parent(population, fitness, rng)
            b = select_parent(population, fitness, rng)
            child = crossover(a, b, vehicles_by_id, rng)
            if rng.random() < MUTATION_PROBABILITY:
                child = mutate(child, ship, vehicles_by_id, rng)
            new_population.append(child)

        population = new_population
        fitness = evaluate_population(population, ship, vehicles_by_id)
        best_state, best_value = update_best_feasible(population, fitness, ship, vehicles_by_id, best_state, best_value)
        history.append(max(fitness))
        history_avg.append(sum(fitness) / len(fitness))


    if not is_feasible(best_state, ship, vehicles_by_id):
        raise RuntimeError("best_state not feasible (bug)")
    final_value = objective(best_state, ship, vehicles_by_id)
    duration = perf_counter() - start_time

    return SearchResult(
        initial_state=initial_result,
        final_state=best_state,
        final_value=final_value,
        history=history,
        iterations=max_iter,
        duration_s=duration,
        extra={
            "history_avg": history_avg,
            "population_size": population_size,
            "max_iter": max_iter,
            "mutation_probability": MUTATION_PROBABILITY,
            "rank_pressure": RANK_PRESSURE,
            "seed": seed,
            "initial_value": initial_value
        },
    )
