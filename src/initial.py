import random
from src.model import Orientation, Placement, Ship, State, Vehicle
from src.neighbors import valid_positions


def random_initial_state(ship:Ship, vehicles_by_id:dict[str, Vehicle], rng:random.Random, p_inside:float | None = None) -> State:
    # Initial state random yang harus feasible: kendaraan hanya dinaikkan if and only if muat di dalam kapal, tidak overlap, dan total berat tetap <= kapasitas.
    if p_inside is None:
        p_inside = default_p_inside(ship, vehicles_by_id)
    if not 0.0 <= p_inside <= 1.0:
        raise ValueError("p_inside harus di antara 0 dan 1")

    occupied: set[tuple[int, int]] = set()
    load = 0
    chosen: dict[str, Placement] = {}

    # Acak order biar rata distribusinya
    order = list(vehicles_by_id)
    rng.shuffle(order)
    for v_id in order:
        v = vehicles_by_id[v_id]
        orientations = rng.sample(list(Orientation), 2)
        placement = Placement(vehicle_id=v_id, x=None, y=None, orientation=orientations[0])

        if rng.random() < p_inside and load + v.weight <= ship.max_capacity:
            for orientation in orientations:
                free = [
                    (x, y)
                    for x, y in valid_positions(ship, v, orientation)
                    if not _overlaps(occupied, x, y, *v.size_taken(orientation))
                ]
                if free:
                    x, y = rng.choice(free)
                    placement = Placement(vehicle_id=v_id, x=x, y=y, orientation=orientation)
                    occupied.update(placement.grids(v))
                    load += v.weight
                    break

        chosen[v_id] = placement

    # Simpan dengan urutan vehicles_by_id (urutan gen GA)
    return State({v_id: chosen[v_id] for v_id in vehicles_by_id})


# Helper buat cek grid yang udah keisi
def _overlaps(occupied: set[tuple[int, int]], x: int, y: int, vw: int, vl: int) -> bool:
    return any((x + i, y + j) in occupied for i in range(vw) for j in range(vl))


# Pilih p supaya rata-rata (expected) total berat & luas di state awal tidak melebihi kapasitas kapal: E[total] = p * total  ->  p = batas / total
def default_p_inside(ship:Ship, vehicles_by_id:dict[str, Vehicle]) -> float:
    total_weight = sum (v.weight for v in vehicles_by_id.values())
    total_area = sum(v.w * v.l for v in vehicles_by_id.values())
    ratios = [1.0]
    if total_weight > 0:
        ratios.append(ship.max_capacity / total_weight)
    if total_area > 0:
        ratios.append((ship.w * ship.l) / total_area)
    return min(ratios)
