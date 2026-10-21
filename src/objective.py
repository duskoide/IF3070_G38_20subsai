from src.model import Ship, State, Vehicle
from collections import Counter

# Penalties
OVERLAP_PENALTY = 1

OUT_OF_BOUNDS_PENALTY = 1

OVERWEIGHT_PENALTY = 1

# Helper functions
# Gunakan vehicles_by_id dari loader untuk lookup kendaraan berdasarkan ID.
def total_fee(state: State, vehicles_by_id: dict[str, Vehicle]) -> int:
    total = 0

    for p in state.placements.values():
        if p.is_inside():
            vehicle = vehicles_by_id[p.vehicle_id]
            total += vehicle.fee

    return total


def total_weight(state: State, vehicles_by_id: dict[str, Vehicle]) -> int:
    total = 0

    for p in state.placements.values():
        if p.is_inside():
            vehicle = vehicles_by_id[p.vehicle_id]
            total += vehicle.weight

    return total


def count_overlap(state: State, vehicles_by_id: dict[str, Vehicle]) -> int:
    occupancy: Counter[tuple[int, int]] = Counter()

    for p in state.placements.values():
        if p.is_inside():
            vehicle = vehicles_by_id[p.vehicle_id]
            occupancy.update(p.grids(vehicle))

    return sum(c - 1 for c in occupancy.values() if c > 1)


def count_out_of_bounds(
    state: State, ship: Ship, vehicles_by_id: dict[str, Vehicle]
) -> int:
    count = 0
    for p in state.placements.values():
        x, y = p.x, p.y
        if x is not None and y is not None:
            vehicle = vehicles_by_id[p.vehicle_id]
            vw, vl = vehicle.size_taken(p.orientation)
            if not ship.in_boundary(x, y, vw, vl):
                count += 1

    return count


def excess_weight(state: State, ship: Ship, vehicles_by_id: dict[str, Vehicle]) -> int:
    w = total_weight(state, vehicles_by_id)
    return max(0, w - ship.max_capacity)


# Objective function
def objective(state: State, ship: Ship, vehicles_by_id: dict[str, Vehicle]) -> float:
    fee = total_fee(state, vehicles_by_id)
    overlap = count_overlap(state, vehicles_by_id)
    out_of_bounds = count_out_of_bounds(state, ship, vehicles_by_id)
    overweight = excess_weight(state, ship, vehicles_by_id)

    return float(
        fee
        - overlap * OVERLAP_PENALTY
        - out_of_bounds * OUT_OF_BOUNDS_PENALTY
        - overweight * OVERWEIGHT_PENALTY
    )

# Checker untuk state yang feasible; i.e. tidak melanggar constraints
def is_feasible(state: State, ship: Ship, vehicles_by_id: dict[str, Vehicle]) -> bool:
    return (
        excess_weight(state, ship, vehicles_by_id) == 0
        and count_out_of_bounds(state, ship, vehicles_by_id) == 0
        and count_overlap(state, vehicles_by_id) == 0
    )
