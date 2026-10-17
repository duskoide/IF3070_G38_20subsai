from dataclasses import dataclass, replace
from enum import Enum


class Orientation(str, Enum):
    HORIZONTAL = "Horizontal"
    VERTICAL = "Vertical"

    def flipped(self):
        return (
            Orientation.VERTICAL
            if self is Orientation.HORIZONTAL
            else Orientation.HORIZONTAL
        )


@dataclass(frozen=True)
class Vehicle:
    id: str
    w: int  # patokan/orientasi lebar kendaraan: jarak dari ban kiri ke ban kendaraan
    l: int  # patokan/orientasi panjang kendaraan: jarak dari bagian depan ke bagian belakang mobil
    fee: int
    weight: int
    eta: int

    def size_taken(
        self, orientation: Orientation
    ) -> tuple[int, int]:  # expected output 2 buah int.
        if orientation is Orientation.VERTICAL:
            return (self.l, self.w)
        return (self.w, self.l)


@dataclass(frozen=True)
class Ship:
    w: int
    l: int
    max_capacity: int

    def in_boundary(self, x: int, y: int, vw: int, vl: int) -> bool:
        # x & y = koordinat posisi kendaraan diletakkan
        # vw & vl = lebar dan panjang (respectively) kendaraan
        return x >= 0 and y >= 0 and x + vw <= self.w and y + vl <= self.l

    def within_capacity(self, total_weight: int) -> bool:
        return self.max_capacity >= total_weight


@dataclass(frozen=True)
class Placement:
    vehicle_id: str
    x: int | None  # none = kendaraan ngga di kapal
    y: int | None
    orientation: Orientation  # vertical or horizontal

    def rotated(self):
        return replace(self, orientation=self.orientation.flipped())

    # fungsi checker apakah placement kendaraan ada di dalam kapal
    def is_inside(self):
        return self.x is not None and self.y is not None

    # fungsi untuk output grid yang di-occupy oleh kendaraan tertentu dengan placement p
    def cells(self, vehicle: Vehicle):
        if not self.is_inside():
            return []
        vw, vl = vehicle.size_taken(self.orientation)
        return [
            (self.x + i, self.y + j) for i in range(vw) for j in range(vl)
        ]  # misal: (1,2), (1,3), (2,2), (2,3)


@dataclass(frozen=True)
class State:
    placements: dict[str, Placement]  # lookup table key->value
