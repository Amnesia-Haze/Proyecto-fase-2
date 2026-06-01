"""test_instances.py — Tests para VRPInstance, I/O CSV y generate_cvrp_instance.

Cubre los requisitos:
  R1. Lectura correcta de CSV.
  R2. Rechazo de CSV sin columnas obligatorias.
  R3. Rechazo de demandas negativas.
  R4. Rechazo de q_i > Q.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.models.instances import (
    VRPInstance,
    generate_cvrp_instance,
    read_instance_csv,
    write_instance_csv,
)

# ---------------------------------------------------------------------------
# R1 — Lectura correcta de CSV
# ---------------------------------------------------------------------------

class TestCSVRoundtrip:
    """Un CSV escrito con write_instance_csv se lee idénticamente."""

    def test_coords_preserved(self, tiny_csv: Path, tiny_instance: VRPInstance):
        inst = read_instance_csv(tiny_csv)
        np.testing.assert_allclose(inst.coords, tiny_instance.coords, atol=1e-4,
                                   err_msg="coords no coinciden tras roundtrip")

    def test_demands_preserved(self, tiny_csv: Path, tiny_instance: VRPInstance):
        inst = read_instance_csv(tiny_csv)
        np.testing.assert_allclose(inst.demands, tiny_instance.demands, atol=1e-4,
                                   err_msg="demands no coinciden tras roundtrip")

    def test_capacity_preserved(self, tiny_csv: Path, tiny_instance: VRPInstance):
        inst = read_instance_csv(tiny_csv)
        assert inst.capacity == pytest.approx(tiny_instance.capacity)

    def test_name_preserved(self, tiny_csv: Path, tiny_instance: VRPInstance):
        inst = read_instance_csv(tiny_csv)
        assert inst.name == tiny_instance.name

    def test_n_customers_preserved(self, tiny_csv: Path, tiny_instance: VRPInstance):
        inst = read_instance_csv(tiny_csv)
        assert inst.n_customers == tiny_instance.n_customers

    def test_n_nodes_preserved(self, tiny_csv: Path, tiny_instance: VRPInstance):
        inst = read_instance_csv(tiny_csv)
        assert inst.n_nodes == tiny_instance.n_nodes

    def test_depot_demand_is_zero(self, tiny_csv: Path):
        inst = read_instance_csv(tiny_csv)
        assert inst.demands[0] == 0.0

    def test_file_not_found_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            read_instance_csv(tmp_path / "nonexistent.csv")

    def test_nodes_sorted_by_index(self, tmp_path: Path, tiny_instance: VRPInstance):
        """El lector debe ordenar filas por 'node' sin importar el orden del CSV."""
        path = tmp_path / "shuffled.csv"
        n = tiny_instance.n_nodes
        df = pd.DataFrame({
            "node": list(range(n))[::-1],   # orden invertido
            "x": tiny_instance.coords[:, 0][::-1],
            "y": tiny_instance.coords[:, 1][::-1],
            "demand": tiny_instance.demands[::-1],
            "capacity": tiny_instance.capacity,
            "name": tiny_instance.name,
        })
        df.to_csv(path, index=False)
        inst = read_instance_csv(path)
        np.testing.assert_allclose(inst.coords, tiny_instance.coords, atol=1e-4)


# ---------------------------------------------------------------------------
# R2 — Rechazo de CSV sin columnas obligatorias
# ---------------------------------------------------------------------------

class TestCSVMissingColumns:
    """CSV sin alguna columna obligatoria → ValueError."""

    REQUIRED_COLUMNS = ["node", "x", "y", "demand", "capacity"]

    def _write_csv_without(self, path: Path, drop_col: str, instance: VRPInstance):
        n = instance.n_nodes
        df = pd.DataFrame({
            "node": range(n),
            "x": instance.coords[:, 0],
            "y": instance.coords[:, 1],
            "demand": instance.demands,
            "capacity": instance.capacity,
            "name": instance.name,
        }).drop(columns=[drop_col])
        df.to_csv(path, index=False)

    @pytest.mark.parametrize("missing_col", REQUIRED_COLUMNS)
    def test_missing_column_raises(
        self, tmp_path: Path, tiny_instance: VRPInstance, missing_col: str
    ):
        path = tmp_path / f"missing_{missing_col}.csv"
        self._write_csv_without(path, missing_col, tiny_instance)
        with pytest.raises(ValueError, match=missing_col):
            read_instance_csv(path)

    def test_missing_depot_raises(self, tmp_path: Path, tiny_instance: VRPInstance):
        """CSV sin el nodo 0 (depósito) debe rechazarse."""
        path = tmp_path / "no_depot.csv"
        n = tiny_instance.n_nodes
        df = pd.DataFrame({
            "node": range(1, n),          # nodo 0 omitido
            "x": tiny_instance.coords[1:, 0],
            "y": tiny_instance.coords[1:, 1],
            "demand": tiny_instance.demands[1:],
            "capacity": tiny_instance.capacity,
            "name": tiny_instance.name,
        })
        df.to_csv(path, index=False)
        with pytest.raises(ValueError):
            read_instance_csv(path)

    def test_duplicate_node_ids_raises(self, tmp_path: Path, tiny_instance: VRPInstance):
        """CSV con nodos duplicados debe rechazarse."""
        path = tmp_path / "duplicate_nodes.csv"
        n = tiny_instance.n_nodes
        nodes = list(range(n))
        nodes[-1] = nodes[-2]            # duplicar último nodo
        df = pd.DataFrame({
            "node": nodes,
            "x": tiny_instance.coords[:, 0],
            "y": tiny_instance.coords[:, 1],
            "demand": tiny_instance.demands,
            "capacity": tiny_instance.capacity,
        })
        df.to_csv(path, index=False)
        with pytest.raises(ValueError):
            read_instance_csv(path)


# ---------------------------------------------------------------------------
# R3 — Rechazo de demandas negativas
# ---------------------------------------------------------------------------

class TestNegativeDemands:
    """VRPInstance rechaza demandas negativas en cualquier cliente."""

    def _make_coords(self, n: int) -> np.ndarray:
        return np.zeros((n, 2))

    def test_single_negative_demand(self):
        coords = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        demands = np.array([0.0, -1.0, 5.0])      # C1 negativa
        with pytest.raises(ValueError, match="negativa"):
            VRPInstance(coords=coords, demands=demands, capacity=10.0)

    def test_all_negative_demands(self):
        coords = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        demands = np.array([0.0, -3.0, -5.0])
        with pytest.raises(ValueError, match="negativa"):
            VRPInstance(coords=coords, demands=demands, capacity=10.0)

    def test_small_negative_demand(self):
        """Demanda de -0.001 también debe rechazarse."""
        coords = np.array([[0.0, 0.0], [1.0, 0.0]])
        demands = np.array([0.0, -0.001])
        with pytest.raises(ValueError, match="negativa"):
            VRPInstance(coords=coords, demands=demands, capacity=10.0)

    def test_depot_nonzero_demand_raises(self):
        """La demanda del depósito debe ser exactamente 0."""
        coords = np.array([[0.0, 0.0], [1.0, 0.0]])
        demands = np.array([1.0, 5.0])             # depósito != 0
        with pytest.raises(ValueError, match="depósito"):
            VRPInstance(coords=coords, demands=demands, capacity=10.0)

    def test_negative_depot_demand_raises(self):
        coords = np.array([[0.0, 0.0], [1.0, 0.0]])
        demands = np.array([-2.0, 5.0])
        with pytest.raises(ValueError):
            VRPInstance(coords=coords, demands=demands, capacity=10.0)

    def test_zero_customer_demand_is_valid(self):
        """Una demanda de 0 en un cliente es inusual pero válida (no negativa)."""
        coords = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        demands = np.array([0.0, 0.0, 5.0])
        inst = VRPInstance(coords=coords, demands=demands, capacity=10.0)
        assert inst.demands[1] == 0.0


# ---------------------------------------------------------------------------
# R4 — Rechazo de q_i > Q
# ---------------------------------------------------------------------------

class TestDemandExceedsCapacity:
    """VRPInstance rechaza cualquier cliente cuya demanda supere Q."""

    def test_single_customer_exceeds(self):
        coords = np.array([[0.0, 0.0], [1.0, 0.0]])
        demands = np.array([0.0, 60.0])
        with pytest.raises(ValueError, match="capacidad"):
            VRPInstance(coords=coords, demands=demands, capacity=50.0)

    def test_one_of_many_exceeds(self):
        coords = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        demands = np.array([0.0, 10.0, 55.0, 20.0])   # C2 excede Q=50
        with pytest.raises(ValueError, match="capacidad"):
            VRPInstance(coords=coords, demands=demands, capacity=50.0)

    def test_demand_equal_to_capacity_is_valid(self):
        """q_i == Q es válido: el cliente puede ser servido solo."""
        coords = np.array([[0.0, 0.0], [1.0, 0.0]])
        demands = np.array([0.0, 50.0])
        inst = VRPInstance(coords=coords, demands=demands, capacity=50.0)
        assert inst.demands[1] == 50.0

    def test_capacity_must_be_positive(self):
        coords = np.array([[0.0, 0.0], [1.0, 0.0]])
        demands = np.array([0.0, 5.0])
        with pytest.raises(ValueError):
            VRPInstance(coords=coords, demands=demands, capacity=0.0)

    def test_negative_capacity_raises(self):
        coords = np.array([[0.0, 0.0], [1.0, 0.0]])
        demands = np.array([0.0, 5.0])
        with pytest.raises(ValueError):
            VRPInstance(coords=coords, demands=demands, capacity=-10.0)


# ---------------------------------------------------------------------------
# Validaciones adicionales de VRPInstance
# ---------------------------------------------------------------------------

class TestVRPInstanceValidation:

    def test_too_few_nodes_raises(self):
        """Solo el depósito (N=1) debe rechazarse."""
        coords = np.array([[0.0, 0.0]])
        demands = np.array([0.0])
        with pytest.raises(ValueError):
            VRPInstance(coords=coords, demands=demands, capacity=10.0)

    def test_wrong_coords_shape_raises(self):
        with pytest.raises(ValueError, match="shape"):
            VRPInstance(
                coords=np.array([0.0, 1.0, 2.0]),   # 1D, no (N,2)
                demands=np.array([0.0, 5.0]),
                capacity=10.0,
            )

    def test_coords_demands_length_mismatch(self):
        coords = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])   # N=3
        demands = np.array([0.0, 5.0])                              # len=2
        with pytest.raises(ValueError):
            VRPInstance(coords=coords, demands=demands, capacity=10.0)


# ---------------------------------------------------------------------------
# generate_cvrp_instance
# ---------------------------------------------------------------------------

class TestGenerateCVRP:

    def test_reproducibility(self):
        a = generate_cvrp_instance(n_customers=10, capacity=50, seed=99)
        b = generate_cvrp_instance(n_customers=10, capacity=50, seed=99)
        np.testing.assert_array_equal(a.coords, b.coords)
        np.testing.assert_array_equal(a.demands, b.demands)

    def test_different_seeds_differ(self):
        a = generate_cvrp_instance(n_customers=10, capacity=50, seed=1)
        b = generate_cvrp_instance(n_customers=10, capacity=50, seed=2)
        assert not np.array_equal(a.coords, b.coords)

    def test_demands_within_range(self):
        inst = generate_cvrp_instance(
            n_customers=20, capacity=50, demand_range=(2.0, 15.0), seed=0
        )
        assert float(inst.demands[1:].min()) >= 2.0
        assert float(inst.demands[1:].max()) <= 15.0

    def test_depot_demand_zero(self):
        inst = generate_cvrp_instance(n_customers=5, capacity=30, seed=0)
        assert inst.demands[0] == 0.0

    def test_n_customers_correct(self):
        inst = generate_cvrp_instance(n_customers=12, capacity=40, seed=0)
        assert inst.n_customers == 12

    def test_invalid_n_customers_raises(self):
        with pytest.raises(ValueError):
            generate_cvrp_instance(n_customers=0, capacity=50, seed=0)

    def test_invalid_capacity_raises(self):
        with pytest.raises(ValueError):
            generate_cvrp_instance(n_customers=5, capacity=-10, seed=0)

    def test_demand_clipped_to_capacity(self):
        """demand_range máximo > Q debe recortarse automáticamente."""
        inst = generate_cvrp_instance(
            n_customers=30, capacity=10, demand_range=(1.0, 999.0), seed=0
        )
        assert float(inst.demands[1:].max()) <= 10.0
