from src.model import Ship, Vehicle
import json


def load_problem(path: str) -> tuple[Ship, list[Vehicle]]:
    with open(path) as f:
        data = json.load(f)

    ship = Ship(**data["ship"])

    vehicles = [Vehicle(**v) for v in data["vehicles"]]

    # dupe-id checker
    seen = set()
    for v in vehicles:
        if v.id in seen:
            raise ValueError(f"Duplicate vehicle id: {v.id!r}")
        seen.add(v.id)

    return (ship, vehicles)


if __name__ == "__main__":
    ship, vehicles = load_problem("src/data/sample.json")
    print(ship)
    print(len(vehicles))
    for v in vehicles:
        print(v)
