import random
from src.model import Orientation, Placement, Ship, State, Vehicle
from src.neighbors import valid_positions

def random_initial_state(ship:Ship, vehicles_by_id:dict[str, Vehicle], rng:random.Random, p_inside:float | None = None) -> State:
    if p_inside is None:
        p_inside = default_p_inside(ship, vehicles_by_id)
    placements = {}
    for v in vehicles_by_id.values():
        orientation = rng.choice(list(Orientation))
        positions = list (valid_positions(ship, v, orientation))
        if positions and rng.random() < p_inside:
            x, y = rng.choice(positions)
        else:
            x, y = None, None
        placements[v.id] = Placement(vehicle_id=v.id, x=x, y=y, orientation=orientation)
    return State(placements)

# Pilih p supaya rata-rata (expected) total berat & luas di state awal
# tidak melebihi kapasitas kapal: E[total] = p * total  ->  p = batas / total
def default_p_inside(ship:Ship, vehicles_by_id:dict[str, Vehicle]) -> float:
    total_weight = sum (v.weight for v in vehicles_by_id.values())
    total_area = sum(v.w * v.l for v in vehicles_by_id.values())
    ratios = [1.0]
    if total_weight > 0:
        ratios.append(ship.max_capacity / total_weight)
    if total_area > 0:
        ratios.append((ship.w * ship.l) / total_area)
    return min(ratios)