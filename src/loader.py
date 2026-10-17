from src.model import Ship, Vehicle
import json


def load_problem(path: str) -> tuple[Ship, list[Vehicle]]:
    with open(path) as f:
        data = json.load(f)

    ship = Ship(**data["ship"])

    vehicles = [Vehicle(**v) for v in data["vehicles"]]

    return (ship, vehicles)


if __name__ == "__main__":
    ship, vehicles = load_problem("src/data/sample.json")
    print(ship)
    print(len(vehicles))
    for v in vehicles:
        print(v)
