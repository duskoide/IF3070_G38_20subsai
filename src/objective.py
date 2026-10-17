from src.model import Ship, State, Vehicle

OVERLAP_PENALTY = None  # per overlapping grid
OUT_OF_BOUNDS_PENALTY = None  # per vehicle yang di luar kapal
OVERWEIGHT_PENALTY = None  # per satuan unit di atas max_capacity


def total_fee(state: State, vehicles_by_id: dict[str, Vehicle]) -> int:
    # Total fee untuk seluruh kendaraan di atas kapal
    raise NotImplementedError


def total_weight(state: State, vehicles_by_id: dict[str, Vehicle]) -> int:
    # Total fee untuk seluruh kendaraan di atas kapal
    raise NotImplementedError


def count_overlap(state: State, vehicles_by_id: dict[str, Vehicle]) -> int:
    # Jumlah grid yang di-occupy oleh lebih dari satu kendaraan
    raise NotImplementedError


def count_out_of_bounds(
    state: State, ship: Ship, vehicles_by_id: dict[str, Vehicle]
) -> int:
    # Jumlah kendaraan yang tidak sepenuhnya berada di dalam kapal
    raise NotImplementedError


def overweight(state: State, ship: Ship, vehicles_by_id: dict[str, Vehicle]) -> int:
    # Total berat yang melebihi max_capacity
    raise NotImplementedError


def objective(state: State, ship: Ship, vehicles: list[Vehicle]) -> float:
    # Total fee dikurangi penalty, nanti dicall buat algorithm
    raise NotImplementedError
