from src.model import Ship, Vehicle
import json


def load_problem(path: str) -> tuple[Ship, dict[str, Vehicle]]:
    with open(path) as f:
        data = json.load(f)

    ship = Ship(**data["ship"])

    # Dict mempertahankan urutan input; tolak ID duplikat sebelum menimpa data.
    vehicles_by_id: dict[str, Vehicle] = {}
    for raw_vehicle in data["vehicles"]:
        vehicle = Vehicle(**raw_vehicle)
        if vehicle.id in vehicles_by_id:
            raise ValueError(f"Duplicate vehicle id: {vehicle.id!r}")
        vehicles_by_id[vehicle.id] = vehicle

    return (ship, vehicles_by_id)


if __name__ == "__main__":
    ship, vehicles_by_id = load_problem("src/data/sample.json")
    print(ship)
    print(len(vehicles_by_id))
    for v in vehicles_by_id.values():
        print(v)
