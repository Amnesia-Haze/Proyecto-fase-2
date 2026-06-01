"""test_solution.py — Tests para funciones puras y VRPSolution.

Cubre los requisitos:
  R5. Costo correcto de una ruta simple.
  R7. Consistencia entre routes_dataframe y arcs_dataframe.

Geometría de tiny_instance (ver conftest.py):
  dist(0,1)=3  dist(1,2)=4  dist(2,3)=3
  dist(0,2)=5  dist(0,3)=4  dist(1,3)=5
"""

from __future__ import annotations

import numpy as np
import pytest

from app.models.instances import VRPInstance
from app.models.solution import (
    VRPSolution,
    check_feasibility,
    route_cost,
    route_load,
    solution_cost,
)


# ---------------------------------------------------------------------------
# Fixtures locales
# ---------------------------------------------------------------------------

@pytest.fixture()
def tiny_solution(tiny_instance: VRPInstance) -> VRPSolution:
    """Solución manual factible para tiny_instance (Q=20).

    Ruta 1: [1, 2]  carga=13  costo=3+4+5=12
    Ruta 2: [3]     carga=6   costo=4+4=8
    """
    return VRPSolution(
        instance=tiny_instance,
        routes=[[1, 2], [3]],
        method="Manual",
    )


# ---------------------------------------------------------------------------
# R5 — Costo correcto de una ruta simple
# ---------------------------------------------------------------------------

class TestRouteCost:
    """Verifica route_cost con geometría exacta (sin error de punto flotante)."""

    def test_single_node_route(self, tiny_instance: VRPInstance):
        # Ruta [1]: 0→C1→0 = 3 + 3 = 6
        assert route_cost([1], tiny_instance) == pytest.approx(6.0)

    def test_two_node_route(self, tiny_instance: VRPInstance):
        # Ruta [1,2]: 0→C1→C2→0 = 3 + 4 + 5 = 12
        assert route_cost([1, 2], tiny_instance) == pytest.approx(12.0)

    def test_three_node_route(self, tiny_instance: VRPInstance):
        # Ruta [1,2,3]: 0→C1→C2→C3→0 = 3 + 4 + 3 + 4 = 14
        assert route_cost([1, 2, 3], tiny_instance) == pytest.approx(14.0)

    def test_empty_route_is_zero(self, tiny_instance: VRPInstance):
        assert route_cost([], tiny_instance) == 0.0

    def test_precomputed_D_gives_same_result(self, tiny_instance: VRPInstance):
        D = tiny_instance.distance_matrix()
        cost_inline = route_cost([1, 2], tiny_instance)
        cost_precomputed = route_cost([1, 2], tiny_instance, D)
        assert cost_inline == pytest.approx(cost_precomputed)

    def test_cost_not_symmetric_on_different_routes(self, tiny_instance: VRPInstance):
        # [2,1] y [1,2] tienen el mismo costo (distancias simétricas)
        cost_12 = route_cost([1, 2], tiny_instance)
        cost_21 = route_cost([2, 1], tiny_instance)
        assert cost_12 == pytest.approx(cost_21)

    def test_route_c3_only(self, tiny_instance: VRPInstance):
        # Ruta [3]: 0→C3→0 = 4 + 4 = 8
        assert route_cost([3], tiny_instance) == pytest.approx(8.0)


class TestRouteLoad:

    def test_single_node_load(self, tiny_instance: VRPInstance):
        # C1 tiene demanda 5.0
        assert route_load([1], tiny_instance) == pytest.approx(5.0)

    def test_multi_node_load(self, tiny_instance: VRPInstance):
        # C1=5 + C2=8 = 13
        assert route_load([1, 2], tiny_instance) == pytest.approx(13.0)

    def test_empty_route_load(self, tiny_instance: VRPInstance):
        assert route_load([], tiny_instance) == 0.0


class TestSolutionCost:

    def test_solution_cost_additive(self, tiny_instance: VRPInstance):
        """solution_cost debe igualar la suma de route_cost individuales."""
        routes = [[1, 2], [3]]
        expected = route_cost([1, 2], tiny_instance) + route_cost([3], tiny_instance)
        assert solution_cost(routes, tiny_instance) == pytest.approx(expected)

    def test_solution_cost_single_route(self, tiny_instance: VRPInstance):
        assert solution_cost([[1, 2, 3]], tiny_instance) == pytest.approx(14.0)

    def test_solution_cost_ignores_empty_routes(self, tiny_instance: VRPInstance):
        cost_with = solution_cost([[1, 2], [3]], tiny_instance)
        cost_without = solution_cost([[1, 2], [], [3]], tiny_instance)
        assert cost_with == pytest.approx(cost_without)


# ---------------------------------------------------------------------------
# Factibilidad
# ---------------------------------------------------------------------------

class TestCheckFeasibility:

    def test_feasible_solution(self, tiny_instance: VRPInstance):
        ok, errors = check_feasibility([[1, 2], [3]], tiny_instance)
        assert ok
        assert errors == []

    def test_single_route_all_customers(self, tiny_instance: VRPInstance):
        # Q=20, carga total=5+8+6=19 ≤ 20 → factible
        ok, errors = check_feasibility([[1, 2, 3]], tiny_instance)
        assert ok

    def test_infeasible_duplicate_customer(self, tiny_instance: VRPInstance):
        ok, errors = check_feasibility([[1, 2], [2, 3]], tiny_instance)
        assert not ok
        assert any("más de una vez" in e for e in errors)

    def test_infeasible_missing_customer(self, tiny_instance: VRPInstance):
        ok, errors = check_feasibility([[1, 2]], tiny_instance)   # C3 falta
        assert not ok
        assert any("no atendidos" in e for e in errors)

    def test_infeasible_capacity_exceeded(self, tiny_instance: VRPInstance):
        # C1+C2+C3 = 19 < Q=20; pero si creamos instancia con Q=10:
        coords = tiny_instance.coords
        demands = tiny_instance.demands
        inst_small_q = VRPInstance(
            coords=coords, demands=np.array([0.0, 5.0, 4.0, 4.0]), capacity=7.0
        )
        # [1,2] = 5+4=9 > Q=7
        ok, errors = check_feasibility([[1, 2], [3]], inst_small_q)
        assert not ok
        assert any("capacidad" in e for e in errors)

    def test_infeasible_invalid_node(self, tiny_instance: VRPInstance):
        ok, errors = check_feasibility([[1, 99], [3]], tiny_instance)
        assert not ok
        assert any("99" in e for e in errors)

    def test_empty_routes_filtered(self, tiny_instance: VRPInstance):
        """Rutas vacías no deben causar errores."""
        ok, errors = check_feasibility([[1, 2], [], [3]], tiny_instance)
        assert ok


# ---------------------------------------------------------------------------
# R7 — Consistencia routes_dataframe / arcs_dataframe
# ---------------------------------------------------------------------------

class TestDataframeConsistency:
    """Verifica que routes_dataframe y arcs_dataframe describen la misma solución."""

    def test_routes_df_shape(self, tiny_solution: VRPSolution):
        df = tiny_solution.to_routes_dataframe()
        assert len(df) == tiny_solution.n_vehicles()

    def test_routes_df_required_columns(self, tiny_solution: VRPSolution):
        df = tiny_solution.to_routes_dataframe()
        required = {"Ruta", "N clientes", "Carga", "Capacidad", "Utilización (%)", "Costo", "Secuencia"}
        assert required.issubset(df.columns)

    def test_arcs_df_required_columns(self, tiny_solution: VRPSolution):
        df = tiny_solution.to_arcs_dataframe()
        required = {"Ruta", "from_node", "to_node", "dist"}
        assert required.issubset(df.columns)

    def test_arcs_df_shape(self, tiny_solution: VRPSolution):
        """Número de arcos = sum(len(route) + 1) para cada ruta."""
        df = tiny_solution.to_arcs_dataframe()
        expected_arcs = sum(len(r) + 1 for r in tiny_solution.routes)
        assert len(df) == expected_arcs

    def test_total_cost_consistent(self, tiny_solution: VRPSolution):
        """La suma de costos en routes_df == suma de distancias en arcs_df."""
        routes_df = tiny_solution.to_routes_dataframe()
        arcs_df = tiny_solution.to_arcs_dataframe()
        assert routes_df["Costo"].sum() == pytest.approx(arcs_df["dist"].sum(), rel=1e-4)

    def test_per_route_cost_consistent(self, tiny_solution: VRPSolution):
        """Para cada ruta r: routes_df["Costo"] == arcs_df["dist"].sum()."""
        routes_df = tiny_solution.to_routes_dataframe()
        arcs_df = tiny_solution.to_arcs_dataframe()
        for _, row in routes_df.iterrows():
            r = int(row["Ruta"])
            arc_cost = arcs_df[arcs_df["Ruta"] == r]["dist"].sum()
            assert row["Costo"] == pytest.approx(arc_cost, rel=1e-4), \
                f"Ruta {r}: routes_df={row['Costo']:.4f} vs arcs_df={arc_cost:.4f}"

    def test_per_route_customer_count_consistent(self, tiny_solution: VRPSolution):
        """routes_df["N clientes"] == número de arcos no-depot por ruta."""
        routes_df = tiny_solution.to_routes_dataframe()
        arcs_df = tiny_solution.to_arcs_dataframe()
        for _, row in routes_df.iterrows():
            r = int(row["Ruta"])
            route_arcs = arcs_df[arcs_df["Ruta"] == r]
            # Arcos internos (from_node != 0 o ambos != 0) = len(route) clientes
            # Cada ruta tiene len(route)+1 arcos total; clientes = total - 1
            n_arcs = len(route_arcs)
            assert row["N clientes"] == n_arcs - 1, \
                f"Ruta {r}: routes_df N_clientes={row['N clientes']} vs arcs_df={n_arcs-1}"

    def test_arc_path_continuity(self, tiny_solution: VRPSolution):
        """El to_node de cada arco debe ser el from_node del siguiente dentro de la ruta."""
        arcs_df = tiny_solution.to_arcs_dataframe()
        for ruta_id in arcs_df["Ruta"].unique():
            arcs = arcs_df[arcs_df["Ruta"] == ruta_id].reset_index(drop=True)
            for i in range(len(arcs) - 1):
                assert arcs.loc[i, "to_node"] == arcs.loc[i + 1, "from_node"], \
                    f"Ruta {ruta_id}, arco {i}: discontinuidad en el camino"

    def test_depot_opens_and_closes_each_route(self, tiny_solution: VRPSolution):
        """Cada ruta debe comenzar y terminar con el nodo 0."""
        arcs_df = tiny_solution.to_arcs_dataframe()
        for ruta_id in arcs_df["Ruta"].unique():
            arcs = arcs_df[arcs_df["Ruta"] == ruta_id].reset_index(drop=True)
            assert arcs.loc[0, "from_node"] == 0, \
                f"Ruta {ruta_id} no comienza en el depósito"
            assert arcs.loc[len(arcs) - 1, "to_node"] == 0, \
                f"Ruta {ruta_id} no termina en el depósito"

    def test_non_depot_nodes_match_routes(self, tiny_solution: VRPSolution):
        """Los nodos no-depósito en arcs_df coinciden con los clientes en routes."""
        arcs_df = tiny_solution.to_arcs_dataframe()
        # from_nodes no-depósito (excluye salida del depósito)
        interior_from = set(arcs_df.loc[arcs_df["from_node"] != 0, "from_node"])
        # Todos los clientes de todas las rutas
        all_customers = {node for route in tiny_solution.routes for node in route}
        assert interior_from == all_customers

    def test_exact_cost_route1(self, tiny_solution: VRPSolution):
        """Ruta 1 [1,2]: costo exacto = 3+4+5 = 12."""
        arcs_df = tiny_solution.to_arcs_dataframe()
        cost_r1 = arcs_df[arcs_df["Ruta"] == 1]["dist"].sum()
        assert cost_r1 == pytest.approx(12.0)

    def test_exact_cost_route2(self, tiny_solution: VRPSolution):
        """Ruta 2 [3]: costo exacto = 4+4 = 8."""
        arcs_df = tiny_solution.to_arcs_dataframe()
        cost_r2 = arcs_df[arcs_df["Ruta"] == 2]["dist"].sum()
        assert cost_r2 == pytest.approx(8.0)

    def test_utilization_consistent_with_load(self, tiny_solution: VRPSolution):
        """Utilización (%) = Carga / Capacidad * 100."""
        df = tiny_solution.to_routes_dataframe()
        for _, row in df.iterrows():
            expected_pct = round(row["Carga"] / row["Capacidad"] * 100, 1)
            assert row["Utilización (%)"] == pytest.approx(expected_pct, abs=0.1)

    def test_sequence_column_format(self, tiny_solution: VRPSolution):
        """Columna Secuencia debe empezar y terminar con '0'."""
        df = tiny_solution.to_routes_dataframe()
        for _, row in df.iterrows():
            seq = row["Secuencia"]
            assert seq.startswith("0"), f"Secuencia no empieza en depósito: {seq}"
            assert seq.endswith("0"), f"Secuencia no termina en depósito: {seq}"
