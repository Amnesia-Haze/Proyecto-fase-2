"""nearest_neighbor.py — Heurística del Vecino Más Cercano para CVRP.

Contrato público:
    nn = NearestNeighbor(start_strategy="depot", tie_break="first")
    solution = nn.solve(instance, seed=None)   # → solution.VRPSolution

Convención de rutas: [1, 4, 7] representa 0 → 1 → 4 → 7 → 0.
La instancia nunca se modifica.
"""

from __future__ import annotations

import time
from typing import Literal, Optional

import numpy as np

from app.models.instances import VRPInstance
from app.models.solution import (
    VRPSolution,
    check_feasibility,
    solution_cost,
)
from app.models.vrp_instance import VRPSolution as _LegacySolution

# ---------------------------------------------------------------------------
# Tipos
# ---------------------------------------------------------------------------

StartStrategy = Literal["depot", "random", "farthest"]
TieBreak = Literal["first", "random", "lowest_demand"]


# ---------------------------------------------------------------------------
# Clase principal
# ---------------------------------------------------------------------------

class NearestNeighbor:
    """Heurística Nearest Neighbor constructiva para CVRP.

    Construye rutas de forma greedy: desde el nodo actual elige siempre el
    cliente no visitado más cercano que quepa en la capacidad restante. Al no
    encontrar más candidatos, cierra la ruta y abre una nueva.

    Parameters
    ----------
    start_strategy : {"depot", "random", "farthest"}
        Cómo seleccionar el primer nodo de cada nueva ruta.

        - ``"depot"``    Greedy estándar desde el depósito (sin pre-selección).
        - ``"random"``   Un cliente no visitado al azar pasa a ser el primero.
        - ``"farthest"`` El cliente no visitado más lejano al depósito es el primero.

    tie_break : {"first", "random", "lowest_demand"}
        Cómo romper empates cuando varios candidatos están exactamente a la
        misma distancia del nodo actual.

        - ``"first"``         Menor índice de nodo (determinista).
        - ``"random"``        Elección uniforme aleatoria entre los empatados.
        - ``"lowest_demand"`` Menor demanda entre los empatados.
    """

    def __init__(
        self,
        start_strategy: StartStrategy = "depot",
        tie_break: TieBreak = "first",
    ) -> None:
        _valid_ss = ("depot", "random", "farthest")
        _valid_tb = ("first", "random", "lowest_demand")
        if start_strategy not in _valid_ss:
            raise ValueError(
                f"start_strategy debe ser uno de {_valid_ss}, "
                f"se recibió {start_strategy!r}."
            )
        if tie_break not in _valid_tb:
            raise ValueError(
                f"tie_break debe ser uno de {_valid_tb}, "
                f"se recibió {tie_break!r}."
            )
        self.start_strategy: StartStrategy = start_strategy
        self.tie_break: TieBreak = tie_break

    # ------------------------------------------------------------------
    # Método principal
    # ------------------------------------------------------------------

    def solve(
        self,
        instance: VRPInstance,
        seed: Optional[int] = None,
    ) -> VRPSolution:
        """Resuelve la instancia CVRP y devuelve una :class:`VRPSolution`.

        Parameters
        ----------
        instance : VRPInstance
            Instancia a resolver. **No se modifica.**
        seed : int, optional
            Semilla para reproducibilidad cuando ``tie_break="random"`` o
            ``start_strategy="random"``. Con ``None`` la ejecución no es
            determinista.

        Returns
        -------
        VRPSolution
            Contiene ``routes``, ``method``, ``metadata`` e ``history`` con
            el log paso a paso de la construcción.
        """
        rng = np.random.default_rng(seed)
        t_start = time.perf_counter()

        D = instance.distance_matrix()   # shape (N, N) — copia inmutable
        n = instance.n_customers
        Q = instance.capacity

        visited = [False] * (n + 1)
        visited[0] = True               # depósito siempre marcado

        routes: list[list[int]] = []
        history: list[dict] = []
        route_idx = 0

        while not all(visited[1:]):
            route_idx += 1
            route: list[int] = []
            capacity_left = Q
            route_cost_acc = 0.0
            step_count = 0
            current = 0                 # siempre partimos desde el depósito

            unvisited = [j for j in range(1, n + 1) if not visited[j]]

            # --- Selección del primer nodo según start_strategy ---
            if self.start_strategy == "depot":
                first_node = 0          # greedy puro: empieza en depósito
            elif self.start_strategy == "random":
                first_node = int(rng.choice(unvisited))
            else:                       # "farthest"
                first_node = int(max(unvisited, key=lambda j: D[0, j]))

            history.append({
                "event": "route_open",
                "route": route_idx,
                "start_node": first_node,
            })

            # Si el primer nodo es un cliente, visitarlo directamente
            if first_node != 0:
                dist = float(D[0, first_node])
                visited[first_node] = True
                route.append(first_node)
                capacity_left -= instance.demands[first_node]
                route_cost_acc += dist
                step_count += 1
                history.append(self._step_entry(
                    route_idx, step_count,
                    from_node=0, to_node=first_node,
                    dist=dist,
                    load_after=Q - capacity_left,
                    capacity_left=capacity_left,
                    route_cost_so_far=route_cost_acc,
                ))
                current = first_node

            # --- Extensión greedy ---
            while True:
                candidates = [
                    j for j in range(1, n + 1)
                    if not visited[j] and instance.demands[j] <= capacity_left
                ]
                if not candidates:
                    break

                dists = D[current, candidates]
                min_dist = float(dists.min())

                # Todos los empatados
                tied = [
                    j for j, d in zip(candidates, dists)
                    if abs(float(d) - min_dist) < 1e-10
                ]

                next_node = self._break_tie(tied, rng, instance)

                dist = float(D[current, next_node])
                visited[next_node] = True
                route.append(next_node)
                capacity_left -= instance.demands[next_node]
                route_cost_acc += dist
                step_count += 1

                history.append(self._step_entry(
                    route_idx, step_count,
                    from_node=current, to_node=next_node,
                    dist=dist,
                    load_after=Q - capacity_left,
                    capacity_left=capacity_left,
                    route_cost_so_far=route_cost_acc,
                ))
                current = next_node

            # --- Cerrar ruta: regreso al depósito ---
            if route:
                return_dist = float(D[current, 0])
                route_cost_acc += return_dist
                history.append({
                    "event": "route_close",
                    "route": route_idx,
                    "last_node": current,
                    "return_dist": round(return_dist, 6),
                    "route_total_cost": round(route_cost_acc, 6),
                    "route_load": round(Q - capacity_left, 4),
                    "route_customers": list(route),
                })
                routes.append(route)

        elapsed = round(time.perf_counter() - t_start, 6)

        return VRPSolution(
            instance=instance,
            routes=routes,
            method="Nearest Neighbor",
            metadata={
                "start_strategy": self.start_strategy,
                "tie_break": self.tie_break,
                "seed": seed,
                "computation_time_s": elapsed,
                "n_steps": sum(
                    1 for h in history if h["event"] == "step"
                ),
            },
            history=history,
        )

    # ------------------------------------------------------------------
    # Helpers privados
    # ------------------------------------------------------------------

    def _break_tie(
        self,
        tied: list[int],
        rng: np.random.Generator,
        instance: VRPInstance,
    ) -> int:
        """Aplica la regla tie_break y devuelve el nodo ganador."""
        if len(tied) == 1:
            return tied[0]
        if self.tie_break == "first":
            return min(tied)
        if self.tie_break == "random":
            return int(rng.choice(tied))
        # "lowest_demand"
        return min(tied, key=lambda j: instance.demands[j])

    @staticmethod
    def _step_entry(
        route: int,
        step: int,
        from_node: int,
        to_node: int,
        dist: float,
        load_after: float,
        capacity_left: float,
        route_cost_so_far: float,
    ) -> dict:
        return {
            "event": "step",
            "route": route,
            "step": step,
            "from_node": from_node,
            "to_node": to_node,
            "dist": round(dist, 6),
            "load_after": round(load_after, 4),
            "capacity_left": round(capacity_left, 4),
            "route_cost_so_far": round(route_cost_so_far, 6),
        }

    def __repr__(self) -> str:
        return (
            f"NearestNeighbor("
            f"start_strategy={self.start_strategy!r}, "
            f"tie_break={self.tie_break!r})"
        )


# ---------------------------------------------------------------------------
# Shim de compatibilidad con ALL_ALGORITHMS (devuelve _LegacySolution)
# ---------------------------------------------------------------------------

def solve(instance: VRPInstance) -> _LegacySolution:
    """Wrapper de compatibilidad para ALL_ALGORITHMS y las páginas Streamlit.

    Usa NearestNeighbor con parámetros por defecto y convierte el resultado
    al tipo _LegacySolution que esperan las páginas actuales.
    """
    sol = NearestNeighbor().solve(instance)
    return _LegacySolution(
        instance=instance,
        routes=sol.routes,
        algorithm_name="Nearest Neighbor",
        computation_time=sol.metadata.get("computation_time_s", 0.0),
    )


# ---------------------------------------------------------------------------
# Pruebas
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from app.models.instances import generate_cvrp_instance

    print("=" * 60)
    print("Pruebas — NearestNeighbor")
    print("=" * 60)

    inst = generate_cvrp_instance(n_customers=12, capacity=40, seed=99)
    print(f"\nInstancia: {inst.n_customers} clientes, Q={inst.capacity}")
    print(f"Demandas: {inst.demands[1:].tolist()}")

    # ------------------------------------------------------------------
    # P1. Factibilidad con parámetros por defecto
    # ------------------------------------------------------------------
    nn = NearestNeighbor()
    sol = nn.solve(inst, seed=0)
    ok, errors = sol.feasibility_report()

    print(f"\n[P1] start_strategy=depot, tie_break=first")
    print(f"  Factible : {ok}")
    print(f"  Rutas    : {sol.n_vehicles()}")
    print(f"  Costo    : {sol.total_cost():.4f}")
    assert ok, f"Solución infactible: {errors}"
    assert sol.n_vehicles() > 0
    print("  OK")

    # ------------------------------------------------------------------
    # P2. Cada cliente exactamente una vez
    # ------------------------------------------------------------------
    all_visited = [node for route in sol.routes for node in route]
    assert sorted(all_visited) == list(range(1, inst.n_customers + 1)), \
        f"Clientes faltantes o duplicados: {sorted(all_visited)}"
    print(f"\n[P2] Cobertura completa sin duplicados")
    print("  OK")

    # ------------------------------------------------------------------
    # P3. Ninguna ruta supera Q
    # ------------------------------------------------------------------
    for i, route in enumerate(sol.routes):
        load = sum(inst.demands[j] for j in route)
        assert load <= inst.capacity, \
            f"Ruta {i+1}: carga {load:.2f} > Q={inst.capacity}"
    print(f"\n[P3] Capacidad respetada en todas las rutas")
    print("  OK")

    # ------------------------------------------------------------------
    # P4. La instancia no fue modificada
    # ------------------------------------------------------------------
    inst2 = generate_cvrp_instance(n_customers=12, capacity=40, seed=99)
    assert np.array_equal(inst.coords, inst2.coords)
    assert np.array_equal(inst.demands, inst2.demands)
    print(f"\n[P4] Instancia no modificada")
    print("  OK")

    # ------------------------------------------------------------------
    # P5. Reproducibilidad con seed
    # ------------------------------------------------------------------
    nn_rand = NearestNeighbor(start_strategy="random", tie_break="random")
    sol_a = nn_rand.solve(inst, seed=7)
    sol_b = nn_rand.solve(inst, seed=7)
    assert sol_a.routes == sol_b.routes, "Seed no garantiza reproducibilidad"
    print(f"\n[P5] Reproducibilidad con seed=7")
    print(f"  Costo sol_a={sol_a.total_cost():.4f}  sol_b={sol_b.total_cost():.4f}")
    print("  OK")

    # ------------------------------------------------------------------
    # P6. start_strategy="random" — factible
    # ------------------------------------------------------------------
    sol_r = NearestNeighbor(start_strategy="random").solve(inst, seed=42)
    ok_r, err_r = sol_r.feasibility_report()
    print(f"\n[P6] start_strategy=random")
    print(f"  Factible: {ok_r}  Rutas: {sol_r.n_vehicles()}  Costo: {sol_r.total_cost():.4f}")
    assert ok_r, f"Solución infactible: {err_r}"
    print("  OK")

    # ------------------------------------------------------------------
    # P7. start_strategy="farthest" — factible
    # ------------------------------------------------------------------
    sol_f = NearestNeighbor(start_strategy="farthest").solve(inst, seed=0)
    ok_f, err_f = sol_f.feasibility_report()
    print(f"\n[P7] start_strategy=farthest")
    print(f"  Factible: {ok_f}  Rutas: {sol_f.n_vehicles()}  Costo: {sol_f.total_cost():.4f}")
    assert ok_f, f"Solución infactible: {err_f}"
    print("  OK")

    # ------------------------------------------------------------------
    # P8. tie_break="lowest_demand" — factible
    # ------------------------------------------------------------------
    sol_ld = NearestNeighbor(tie_break="lowest_demand").solve(inst, seed=0)
    ok_ld, _ = sol_ld.feasibility_report()
    print(f"\n[P8] tie_break=lowest_demand")
    print(f"  Factible: {ok_ld}  Costo: {sol_ld.total_cost():.4f}")
    assert ok_ld
    print("  OK")

    # ------------------------------------------------------------------
    # P9. History bien formado
    # ------------------------------------------------------------------
    events = [h["event"] for h in sol.history]
    assert "route_open" in events
    assert "step" in events
    assert "route_close" in events

    # route_open siempre precede a su primer step
    for i, entry in enumerate(sol.history):
        if entry["event"] == "route_open":
            next_event = sol.history[i + 1]["event"] if i + 1 < len(sol.history) else None
            assert next_event in ("step", "route_close"), \
                f"Evento inesperado tras route_open: {next_event}"

    # route_close registra correctamente los clientes
    closes = [h for h in sol.history if h["event"] == "route_close"]
    reconstructed = [c["route_customers"] for c in closes]
    assert reconstructed == sol.routes
    print(f"\n[P9] History bien formado")
    print(f"  Eventos: {dict.fromkeys(events)}  Steps: {sum(1 for e in events if e=='step')}")
    print("  OK")

    # ------------------------------------------------------------------
    # P10. solution_cost coincide con history
    # ------------------------------------------------------------------
    cost_from_history = sum(
        h["route_total_cost"]
        for h in sol.history
        if h["event"] == "route_close"
    )
    assert abs(cost_from_history - sol.total_cost()) < 1e-3, \
        f"Discrepancia: history={cost_from_history:.4f} vs total_cost={sol.total_cost():.4f}"
    print(f"\n[P10] Costo en history == total_cost()")
    print(f"  history sum={cost_from_history:.4f}  total_cost={sol.total_cost():.4f}")
    print("  OK")

    # ------------------------------------------------------------------
    # P11. Parámetros inválidos — deben lanzar ValueError
    # ------------------------------------------------------------------
    try:
        NearestNeighbor(start_strategy="best")
        assert False, "Debía lanzar ValueError"
    except ValueError:
        pass
    try:
        NearestNeighbor(tie_break="min_id")
        assert False, "Debía lanzar ValueError"
    except ValueError:
        pass
    print(f"\n[P11] Parámetros inválidos detectados")
    print("  OK")

    # ------------------------------------------------------------------
    # P12. Instancia de 1 cliente
    # ------------------------------------------------------------------
    inst_tiny = generate_cvrp_instance(n_customers=1, capacity=50, seed=0)
    sol_tiny = NearestNeighbor().solve(inst_tiny, seed=0)
    ok_t, _ = sol_tiny.feasibility_report()
    assert ok_t and sol_tiny.n_vehicles() == 1 and len(sol_tiny.routes[0]) == 1
    print(f"\n[P12] Instancia de 1 cliente")
    print(f"  Rutas: {sol_tiny.routes}  Factible: {ok_t}")
    print("  OK")

    print("\n" + "=" * 60)
    print(f"Todas las pruebas pasaron.  {repr(nn)}")
    print("=" * 60)
