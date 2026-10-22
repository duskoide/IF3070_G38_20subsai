import dataclasses
import random
from time import perf_counter

from src.model import Orientation, Ship, State, Vehicle, Placement
from src.initial import random_initial_state
from src.neighbors import random_neighbor
from src.objective import is_feasible, objective

MUTATION_PROBABILITY = 0.1

MOVE_WEIGHTS = {"swap": 1, "move": 0.2, "move_out": 0.4, "rotate": 0.2}

# k numbers of contestants in select_parent()
TOURNAMENT_SIZE = 3

# Elitism (preserve child superior)
ELITE_COUNT = 2


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

    # tournament selection with k contestants
    n = len(population)
    if n == 0 or n != len(fitness):
        raise ValueError("populasi kosong atau panjang tidak sama dengan fitness")
    if n == 1:
        return population[0]

    k = min(n, TOURNAMENT_SIZE) # handling if population < tournament_size
    tourney_idx = []

    # population & fitness paralel, ambil index fitness. repeating index allowed
    for _ in range (k):
        tourney_idx.append(rng.randrange(n))
    parent_idx: int = max(tourney_idx, key = lambda i: fitness[i]) # index max dari fitness

    return population[parent_idx]

def crossover(
    parent_a: State,
    parent_b: State,
    ship: Ship,
    vehicles_by_id: dict[str, Vehicle],
    rng: random.Random,
) -> State:
    # Acak induk mana yang dapet prioritas area
    if rng.random() < 0.5:
        parent_a, parent_b = parent_b, parent_a

    axis = rng.choice(["x", "y"])
    extent = ship.w if axis == "x" else ship.l
    cut = rng.randint(1, extent - 1) if extent > 1 else extent
    ids = list(vehicles_by_id)
    rng.shuffle(ids)

    # Ambil vehicle A yang keseluruhannya di kiri/atas garis potong
    candidates = []
    for vehicle_id in ids:
        p = parent_a.placements[vehicle_id]
        if not p.is_inside():
            continue
        vw, vl = vehicles_by_id[vehicle_id].size_taken(p.orientation)
        edge = p.x + vw if axis == "x" else p.y + vl
        if edge <= cut:
            candidates.append(p)

    # then try B, terus sisanya A yang masih bisa masuk
    for parent in (parent_b, parent_a):
        for vehicle_id in ids:
            candidates.append(parent.placements[vehicle_id])

    placements = {}
    for vehicle_id in vehicles_by_id:
        p = parent_a.placements[vehicle_id]
        placements[vehicle_id] = dataclasses.replace(p, x=None, y=None)

    occupied = set()
    total_weight = 0
    for p in candidates:
        if not p.is_inside() or placements[p.vehicle_id].is_inside():
            continue
        vehicle = vehicles_by_id[p.vehicle_id]
        vw, vl = vehicle.size_taken(p.orientation)
        if not ship.in_boundary(p.x, p.y, vw, vl):
            continue
        if not ship.within_capacity(total_weight + vehicle.weight):
            continue
        cells = p.grids(vehicle)
        if any(cell in occupied for cell in cells):
            continue

        placements[p.vehicle_id] = p
        occupied.update(cells)
        total_weight += vehicle.weight

    return State(placements)


def mutate(
    child: State, ship: Ship, vehicles_by_id: dict[str, Vehicle], rng: random.Random
) -> State:

    potential_mutation = random_neighbor(child, ship, vehicles_by_id, rng, MOVE_WEIGHTS)
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
    on_iteration=None,  # callback opsional (iterasi, state terbaik generasi, value)
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
    if on_iteration is not None:
        on_iteration(0, population[current_idx], initial_value)

    for i in range(max_iter):
        elite_n = min(ELITE_COUNT, population_size - 1) # count agar populasi 1 masih bisa punya child
        top_child = sorted(range(population_size), key=lambda i: fitness[i], reverse=True)[:elite_n]
        new_population = [State(dict(population[i].placements)) for i in top_child]
        while len(new_population) < population_size:
            a = select_parent(population, fitness, rng)
            b = select_parent(population, fitness, rng)
            child = crossover(a, b, ship, vehicles_by_id, rng)
            if rng.random() < MUTATION_PROBABILITY:
                child = mutate(child, ship, vehicles_by_id, rng)
            new_population.append(child)

        population = new_population
        fitness = evaluate_population(population, ship, vehicles_by_id)
        best_state, best_value = update_best_feasible(population, fitness, ship, vehicles_by_id, best_state, best_value)
        history.append(max(fitness))
        history_avg.append(sum(fitness) / len(fitness))
        if on_iteration is not None:
            current_idx = fitness.index(max(fitness))
            on_iteration(i + 1, population[current_idx], fitness[current_idx])


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
            "tournament_size": TOURNAMENT_SIZE,
            "elite_count": ELITE_COUNT,
            "move_weights": MOVE_WEIGHTS,
            "seed": seed,
            "initial_value": initial_value
        },
    )
