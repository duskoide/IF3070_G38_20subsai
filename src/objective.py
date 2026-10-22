from dataclasses import dataclass

from src.model import Ship, State, Vehicle
from collections import Counter


@dataclass(frozen=True)
class PenaltyWeights:
    overlap: float  # per sel kelebihan okupansi
    out_of_bounds: float  # per kendaraan yang keluar batas
    overweight_fixed: float  # sekali bila total berat > kapasitas
    overweight_rate: float  # per unit berat di atas kapasitas


def penalty_weights(vehicles_by_id: dict[str, Vehicle]) -> PenaltyWeights:
    max_fee = max((v.fee for v in vehicles_by_id.values()), default=0)
    base = max(0, max_fee) + 1
    rate = max(
        (v.fee / v.weight for v in vehicles_by_id.values() if v.weight > 0),
        default=0.0,
    )
    return PenaltyWeights(
        overlap=base,
        out_of_bounds=base,
        overweight_fixed=base,
        overweight_rate=max(0.0, rate),
    )

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

# Helper function untuk bonus objective function #2
def total_eta(state: State, vehicles_by_id: dict[str, Vehicle]) -> int:
    total = 0

    for p in state.placements.values():
        if p.is_inside():
            vehicle = vehicles_by_id[p.vehicle_id]
            total += vehicle.eta

    return total

# Objective function
def objective(state: State, ship: Ship, vehicles_by_id: dict[str, Vehicle]) -> float:
    fee = total_fee(state, vehicles_by_id)
    overlap = count_overlap(state, vehicles_by_id)
    out_of_bounds = count_out_of_bounds(state, ship, vehicles_by_id)
    overweight = excess_weight(state, ship, vehicles_by_id)

    weights = penalty_weights(vehicles_by_id)
    penalty = overlap * weights.overlap + out_of_bounds * weights.out_of_bounds
    if overweight > 0:
        penalty += weights.overweight_fixed + overweight * weights.overweight_rate

    return float(fee - penalty)

# Checker untuk state yang feasible; i.e. tidak melanggar constraints
def is_feasible(state: State, ship: Ship, vehicles_by_id: dict[str, Vehicle]) -> bool:
    return (
        excess_weight(state, ship, vehicles_by_id) == 0
        and count_out_of_bounds(state, ship, vehicles_by_id) == 0
        and count_overlap(state, vehicles_by_id) == 0
    )
