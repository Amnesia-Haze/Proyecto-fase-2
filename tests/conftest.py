"""conftest.py — Fixtures compartidas para todo el suite de pruebas.

Geometría de tiny_instance (triángulos 3-4-5 → distancias exactas sin FP error):

    Depósito (0): (0, 0)
    C1        (1): (3, 0)   dist(0,1) = 3
    C2        (2): (3, 4)   dist(1,2) = 4,  dist(0,2) = 5
    C3        (3): (0, 4)   dist(2,3) = 3,  dist(0,3) = 4

Costos exactos de rutas de referencia:
    [1]       = 3 + 3 = 6
    [2]       = 5 + 5 = 10
    [3]       = 4 + 4 = 8
    [1, 2]    = 3 + 4 + 5 = 12
    [1, 2, 3] = 3 + 4 + 3 + 4 = 14
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.models.instances import VRPInstance, generate_cvrp_instance, write_instance_csv


# ---------------------------------------------------------------------------
# Instancias reutilizables
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def tiny_instance() -> VRPInstance:
    """Instancia de 3 clientes con geometría exacta (triángulos 3-4-5)."""
    coords = np.array([
        [0.0, 0.0],   # depósito
        [3.0, 0.0],   # C1
        [3.0, 4.0],   # C2
        [0.0, 4.0],   # C3
    ])
    demands = np.array([0.0, 5.0, 8.0, 6.0])
    return VRPInstance(coords=coords, demands=demands, capacity=20.0, name="tiny")


@pytest.fixture(scope="session")
def small_instance() -> VRPInstance:
    """Instancia de 10 clientes generada con semilla fija."""
    return generate_cvrp_instance(n_customers=10, capacity=50.0, seed=42)


@pytest.fixture(scope="session")
def medium_instance() -> VRPInstance:
    """Instancia de 25 clientes para pruebas de escala."""
    return generate_cvrp_instance(n_customers=25, capacity=60.0, seed=7)


# ---------------------------------------------------------------------------
# Fixture de archivo CSV temporal
# ---------------------------------------------------------------------------

@pytest.fixture()
def tiny_csv(tmp_path: Path, tiny_instance: VRPInstance) -> Path:
    """Escribe tiny_instance en un CSV temporal y devuelve su ruta."""
    dest = tmp_path / "tiny.csv"
    write_instance_csv(tiny_instance, dest)
    return dest
