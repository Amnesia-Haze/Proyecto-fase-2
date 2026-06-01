"""data_loader.py — Wrappers de conveniencia sobre app.models.instances."""

import pandas as pd

from app.models.instances import (
    VRPInstance,
    generate_cvrp_instance,
    read_instance_csv,
    write_instance_csv,
)


def generate_random_instance(
    num_customers: int,
    vehicle_capacity: float,
    coord_range: tuple = (0, 100),
    demand_range: tuple = (1, 20),
    seed: int = 42,
    name: str | None = None,
) -> VRPInstance:
    """Wrapper de compatibilidad sobre generate_cvrp_instance."""
    return generate_cvrp_instance(
        n_customers=num_customers,
        capacity=vehicle_capacity,
        coord_range=coord_range,
        demand_range=demand_range,
        seed=seed,
        name=name,
    )


def csv_template() -> pd.DataFrame:
    """Plantilla CSV descargable con el formato esperado por read_instance_csv."""
    return pd.DataFrame({
        "node": [0, 1, 2, 3],
        "x": [50.0, 10.5, 45.0, 78.3],
        "y": [50.0, 20.1, 60.7, 35.9],
        "demand": [0.0, 5.0, 8.5, 3.0],
        "capacity": [50.0, 50.0, 50.0, 50.0],
        "name": ["Mi instancia"] * 4,
    })
