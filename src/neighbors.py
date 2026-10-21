import random
import dataclasses

from collections.abc import Iterator
from src.model import Orientation, Placement, Ship, State, Vehicle

# KONSTANTA
MOVE_WEIGHTS = {"swap": 1, "move": 1, "move_out": 1, "rotate": 1}
ALLOW_MOVE_OUT = True
SWAP_ORIENTATION = False

def _with_placements(state: State, *new_placements: Placement) -> State:
    d = dict(state.placements)
    for p in new_placements:
        d[p.vehicle_id] = p
    return State(d)

def valid_positions( 
    ship: Ship, vehicle: Vehicle, orientation: Orientation
) -> Iterator[tuple[int, int]]:
    vw, vl = vehicle.size_taken(orientation)
    if vw > ship.w or vl > ship.l:
        return
    for x in range(0, ship.w - vw + 1):
        for y in range(0, ship.l - vl + 1):
            yield (x, y)

def swap(state: State, a_id: str, b_id: str) -> State | None:
    if a_id == b_id:
        return None
    pa = state.placements[a_id]
    pb = state.placements[b_id]

    if not (pa.is_inside() or pb.is_inside()):
        return None

    pa_baru = dataclasses.replace(pa, x=pb.x, y=pb.y)
    pb_baru = dataclasses.replace(pb, x=pa.x, y=pa.y)
    return _with_placements(state, pa_baru, pb_baru)  

def move(state: State, vehicle_id: str, x: int, y: int) -> State | None:
    p = state.placements[vehicle_id]
    if p.x == x and p.y == y:
        return None
    p_baru = dataclasses.replace(p, x=x, y=y)
    return _with_placements(state, p_baru)

def move_out(state: State, vehicle_id: str) -> State | None:
    p = state.placements[vehicle_id]
    if not p.is_inside():
        return None
    p_baru = dataclasses.replace(p, x=None, y=None)
    return _with_placements(state, p_baru)

def rotate(state: State, vehicle_id: str) -> State | None:
    p = state.placements[vehicle_id]
    if not p.is_inside():
        return None

    return _with_placements(state, p.rotated())

def all_neighbors(
    state: State, ship: Ship, vehicles: list[Vehicle]
) -> Iterator[State]:
    ids = [vehicle.id for vehicle in vehicles]

    for i, a_id in enumerate(ids):
        for b_id in ids[i + 1 :]:
            neighbor = swap(state, a_id, b_id)
            if neighbor is not None:
                yield neighbor

    for vehicle in vehicles:
        placement = state.placements[vehicle.id]
        for x, y in valid_positions(ship, vehicle, placement.orientation):
            neighbor = move(state, vehicle.id, x, y)
            if neighbor is not None:
                yield neighbor

    if ALLOW_MOVE_OUT:
        for vehicle in vehicles:
            if state.placements[vehicle.id].is_inside():
                neighbor = move_out(state, vehicle.id)
                if neighbor is not None:
                    yield neighbor

    for vehicle in vehicles:
        placement = state.placements[vehicle.id]
        if placement.is_inside() and vehicle.w != vehicle.l:
            neighbor = rotate(state, vehicle.id)
            if neighbor is not None:
                yield neighbor

def random_neighbor(
    state: State, ship: Ship, vehicles: list[Vehicle], rng: random.Random
) -> State | None:
    ids = [vehicle.id for vehicle in vehicles]
    move_types = [
        move_type
        for move_type in MOVE_WEIGHTS
        if move_type != "move_out" or ALLOW_MOVE_OUT
    ]
    weights = [MOVE_WEIGHTS[move_type] for move_type in move_types]

    for _ in range(100):
        move_type = rng.choices(move_types, weights=weights, k=1)[0]

        if move_type == "swap":
            if len(ids) < 2:
                continue
            a_id, b_id = rng.sample(ids, 2)
            neighbor = swap(state, a_id, b_id)
        elif move_type == "move":
            if not vehicles:
                continue
            vehicle = rng.choice(vehicles)
            placement = state.placements[vehicle.id]
            positions = [
                position
                for position in valid_positions(ship, vehicle, placement.orientation)
                if position != (placement.x, placement.y)
            ]
            if not positions:
                continue
            x, y = rng.choice(positions)
            neighbor = move(state, vehicle.id, x, y)
        elif move_type == "move_out":
            inside_ids = [
                vehicle_id
                for vehicle_id in ids
                if state.placements[vehicle_id].is_inside()
            ]
            if not inside_ids:
                continue
            neighbor = move_out(state, rng.choice(inside_ids))
        elif move_type == "rotate":
            rotatable_ids = [
                vehicle.id
                for vehicle in vehicles
                if state.placements[vehicle.id].is_inside()
                and vehicle.w != vehicle.l
            ]
            if not rotatable_ids:
                continue
            neighbor = rotate(state, rng.choice(rotatable_ids))
        else:
            continue

        if neighbor is not None:
            return neighbor

    return None