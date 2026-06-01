"""test_nearest_neighbor.py — Tests para NearestNeighbor.

Cubre el requisito:
  R6. Solución factible generada por Nearest Neighbor.

Adicionalmente verifica: cobertura completa, capacidad, inmutabilidad
de la instancia, reproducibilidad, estrategias de inicio, reglas de
desempate y estructura del historial.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.algorithms.heuristics.nearest_neighbor import NearestNeighbor
from app.models.instances import VRPInstance, generate_cvrp_instance
from app.models.solution import VRPSolution, check_feasibility


# ---------------------------------------------------------------------------
# Fixture local: instancia mediana para probar estrategias
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def instance_15() -> VRPInstance:
    return generate_cvrp_instance(n_customers=15, capacity=40, seed=123)


# ---------------------------------------------------------------------------
# R6 — Solución factible generada por Nearest Neighbor
# ---------------------------------------------------------------------------

class TestFeasibility:

    def test_default_params_feasible(self, small_instance: VRPInstance):
        sol = NearestNeighbor().solve(small_instance, seed=0)
        ok, errors = sol.feasibility_report()
        assert ok, f"Solución infactible: {errors}"

    def test_all_customers_served(self, small_instance: VRPInstance):
        sol = NearestNeighbor().solve(small_instance, seed=0)
        served = sorted(node for route in sol.routes for node in route)
        expected = list(range(1, small_instance.n_customers + 1))
        assert served == expected, "Clientes faltantes o duplicados"

    def test_no_customer_duplicated(self, small_instance: VRPInstance):
        sol = NearestNeighbor().solve(small_instance, seed=0)
        served = [node for route in sol.routes for node in route]
        assert len(served) == len(set(served)), "Clientes visitados más de una vez"

    def test_capacity_respected_per_route(self, small_instance: VRPInstance):
        sol = NearestNeighbor().solve(small_instance, seed=0)
        Q = small_instance.capacity
        for i, route in enumerate(sol.routes):
            load = sum(small_instance.demands[j] for j in route)
            assert load <= Q, (
                f"Ruta {i+1}: carga {load:.2f} supera Q={Q}"
            )

    def test_feasible_on_medium_instance(self, medium_instance: VRPInstance):
        sol = NearestNeighbor().solve(medium_instance, seed=0)
        assert sol.is_feasible()

    def test_feasible_with_random_strategy(self, small_instance: VRPInstance):
        sol = NearestNeighbor(start_strategy="random").solve(small_instance, seed=42)
        assert sol.is_feasible()

    def test_feasible_with_farthest_strategy(self, small_instance: VRPInstance):
        sol = NearestNeighbor(start_strategy="farthest").solve(small_instance, seed=0)
        assert sol.is_feasible()

    def test_feasible_with_lowest_demand_tiebreak(self, small_instance: VRPInstance):
        sol = NearestNeighbor(tie_break="lowest_demand").solve(small_instance, seed=0)
        assert sol.is_feasible()

    def test_feasible_single_customer(self):
        inst = generate_cvrp_instance(n_customers=1, capacity=50, seed=0)
        sol = NearestNeighbor().solve(inst, seed=0)
        assert sol.is_feasible()
        assert sol.n_vehicles() == 1
        assert sol.routes == [[1]]


# ---------------------------------------------------------------------------
# Instancia no modificada
# ---------------------------------------------------------------------------

class TestImmutability:

    def test_coords_not_modified(self, small_instance: VRPInstance):
        coords_before = small_instance.coords.copy()
        NearestNeighbor().solve(small_instance, seed=0)
        np.testing.assert_array_equal(small_instance.coords, coords_before)

    def test_demands_not_modified(self, small_instance: VRPInstance):
        demands_before = small_instance.demands.copy()
        NearestNeighbor().solve(small_instance, seed=0)
        np.testing.assert_array_equal(small_instance.demands, demands_before)

    def test_capacity_not_modified(self, small_instance: VRPInstance):
        q_before = small_instance.capacity
        NearestNeighbor().solve(small_instance, seed=0)
        assert small_instance.capacity == q_before


# ---------------------------------------------------------------------------
# Reproducibilidad
# ---------------------------------------------------------------------------

class TestReproducibility:

    def test_same_seed_same_routes(self, instance_15: VRPInstance):
        nn = NearestNeighbor(start_strategy="random", tie_break="random")
        sol_a = nn.solve(instance_15, seed=7)
        sol_b = nn.solve(instance_15, seed=7)
        assert sol_a.routes == sol_b.routes

    def test_different_seeds_may_differ(self, instance_15: VRPInstance):
        nn = NearestNeighbor(start_strategy="random", tie_break="random")
        sol_1 = nn.solve(instance_15, seed=1)
        sol_2 = nn.solve(instance_15, seed=2)
        # No garantizado que difieran, pero con n=15 es casi seguro
        # (test estadístico suave: al menos verificamos que ambas son factibles)
        assert sol_1.is_feasible()
        assert sol_2.is_feasible()

    def test_depot_strategy_is_deterministic_without_seed(
        self, small_instance: VRPInstance
    ):
        """start_strategy=depot + tie_break=first no usa aleatoriedad → mismo resultado."""
        nn = NearestNeighbor(start_strategy="depot", tie_break="first")
        sol_a = nn.solve(small_instance, seed=None)
        sol_b = nn.solve(small_instance, seed=None)
        assert sol_a.routes == sol_b.routes


# ---------------------------------------------------------------------------
# Parámetros: estrategias y desempate
# ---------------------------------------------------------------------------

class TestStrategiesAndTieBreaks:

    @pytest.mark.parametrize("strategy", ["depot", "random", "farthest"])
    def test_start_strategy_produces_feasible(
        self, strategy: str, small_instance: VRPInstance
    ):
        sol = NearestNeighbor(start_strategy=strategy).solve(small_instance, seed=42)
        assert sol.is_feasible(), f"strategy={strategy!r} produjo solución infactible"

    @pytest.mark.parametrize("tb", ["first", "random", "lowest_demand"])
    def test_tie_break_produces_feasible(
        self, tb: str, small_instance: VRPInstance
    ):
        sol = NearestNeighbor(tie_break=tb).solve(small_instance, seed=42)
        assert sol.is_feasible(), f"tie_break={tb!r} produjo solución infactible"

    def test_metadata_records_strategy(self, small_instance: VRPInstance):
        sol = NearestNeighbor(start_strategy="farthest", tie_break="random").solve(
            small_instance, seed=5
        )
        assert sol.metadata["start_strategy"] == "farthest"
        assert sol.metadata["tie_break"] == "random"
        assert sol.metadata["seed"] == 5

    def test_method_name_is_nearest_neighbor(self, small_instance: VRPInstance):
        sol = NearestNeighbor().solve(small_instance, seed=0)
        assert sol.method == "Nearest Neighbor"

    def test_computation_time_recorded(self, small_instance: VRPInstance):
        sol = NearestNeighbor().solve(small_instance, seed=0)
        assert sol.metadata["computation_time_s"] > 0.0

    def test_invalid_start_strategy_raises(self):
        with pytest.raises(ValueError, match="start_strategy"):
            NearestNeighbor(start_strategy="best_first")

    def test_invalid_tie_break_raises(self):
        with pytest.raises(ValueError, match="tie_break"):
            NearestNeighbor(tie_break="min_index")


# ---------------------------------------------------------------------------
# Historial de construcción
# ---------------------------------------------------------------------------

class TestHistory:

    @pytest.fixture()
    def nn_solution(self, small_instance: VRPInstance) -> VRPSolution:
        return NearestNeighbor().solve(small_instance, seed=0)

    def test_history_not_empty(self, nn_solution: VRPSolution):
        assert len(nn_solution.history) > 0

    def test_history_contains_expected_events(self, nn_solution: VRPSolution):
        event_types = {h["event"] for h in nn_solution.history}
        assert event_types == {"route_open", "step", "route_close"}

    def test_route_open_before_steps(self, nn_solution: VRPSolution):
        """Dentro de cada ruta, route_open debe preceder a los steps."""
        current_route = None
        open_seen = False
        for h in nn_solution.history:
            if h["event"] == "route_open":
                current_route = h["route"]
                open_seen = True
            elif h["event"] == "step":
                assert open_seen, "step sin route_open previo"
                assert h["route"] == current_route
            elif h["event"] == "route_close":
                open_seen = False

    def test_route_close_customers_match_routes(self, nn_solution: VRPSolution):
        """route_close["route_customers"] debe coincidir con routes."""
        closes = [h for h in nn_solution.history if h["event"] == "route_close"]
        reconstructed = [c["route_customers"] for c in closes]
        assert reconstructed == nn_solution.routes

    def test_step_count_equals_n_customers(self, nn_solution: VRPSolution):
        """El número de steps debe igualar el total de clientes."""
        steps = [h for h in nn_solution.history if h["event"] == "step"]
        total_customers = sum(len(r) for r in nn_solution.routes)
        assert len(steps) == total_customers

    def test_history_cost_matches_total_cost(self, nn_solution: VRPSolution):
        """La suma de route_total_cost en route_close == total_cost()."""
        history_total = sum(
            h["route_total_cost"]
            for h in nn_solution.history
            if h["event"] == "route_close"
        )
        assert history_total == pytest.approx(nn_solution.total_cost(), rel=1e-4)

    def test_step_capacity_monotonically_decreasing(self, nn_solution: VRPSolution):
        """capacity_left dentro de una ruta nunca debe aumentar."""
        for ruta_id in range(1, nn_solution.n_vehicles() + 1):
            steps = [
                h for h in nn_solution.history
                if h["event"] == "step" and h["route"] == ruta_id
            ]
            caps = [s["capacity_left"] for s in steps]
            assert caps == sorted(caps, reverse=True), \
                f"Ruta {ruta_id}: capacity_left no es monótonamente decreciente"

    def test_no_step_exceeds_capacity(self, nn_solution: VRPSolution):
        """En ningún step capacity_left debe ser negativo."""
        steps = [h for h in nn_solution.history if h["event"] == "step"]
        for step in steps:
            assert step["capacity_left"] >= -1e-9, \
                f"Capacidad negativa en step: {step}"

    def test_history_n_steps_recorded_in_metadata(self, nn_solution: VRPSolution):
        steps_in_history = sum(1 for h in nn_solution.history if h["event"] == "step")
        assert nn_solution.metadata["n_steps"] == steps_in_history
