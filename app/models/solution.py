"""solution.py — Modelo de solución CVRP con métricas, factibilidad y visualización.

Convención de rutas: [1, 4, 7] representa el recorrido 0 → 1 → 4 → 7 → 0,
donde 0 es siempre el depósito. Las rutas vacías se ignoran.

Funciones puras (no requieren instanciar VRPSolution):
    route_load, route_cost, solution_cost, check_feasibility

Clase principal:
    VRPSolution — encapsula rutas, método, metadata e historial.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Optional

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from app.models.instances import VRPInstance


# ---------------------------------------------------------------------------
# Funciones puras
# ---------------------------------------------------------------------------

def route_load(route: list[int], instance: VRPInstance) -> float:
    """Demanda total servida por una ruta.

    Parameters
    ----------
    route : list[int]
        Lista de índices de cliente (1-based). El depósito (0) no se incluye.
    instance : VRPInstance

    Returns
    -------
    float
    """
    return float(sum(instance.demands[i] for i in route))


def route_cost(
    route: list[int],
    instance: VRPInstance,
    D: Optional[np.ndarray] = None,
) -> float:
    """Distancia total de una ruta (depósito → clientes → depósito).

    Parameters
    ----------
    route : list[int]
    instance : VRPInstance
    D : np.ndarray, optional
        Matriz de distancias pre-calculada. Si es None se calcula internamente.

    Returns
    -------
    float
        0.0 para una ruta vacía.
    """
    if not route:
        return 0.0
    if D is None:
        D = instance.distance_matrix()
    cost = D[0, route[0]]
    for a, b in zip(route, route[1:]):
        cost += D[a, b]
    cost += D[route[-1], 0]
    return float(cost)


def solution_cost(routes: list[list[int]], instance: VRPInstance) -> float:
    """Distancia total de todas las rutas.

    Calcula la matriz de distancias una sola vez para eficiencia.

    Parameters
    ----------
    routes : list[list[int]]
    instance : VRPInstance

    Returns
    -------
    float
    """
    D = instance.distance_matrix()
    return sum(route_cost(r, instance, D) for r in routes if r)


def check_feasibility(
    routes: list[list[int]],
    instance: VRPInstance,
) -> tuple[bool, list[str]]:
    """Verifica que las rutas formen una solución CVRP factible.

    Comprobaciones:
    1. Índices de nodo válidos (1 ≤ i ≤ n_customers).
    2. Cada cliente aparece exactamente una vez.
    3. Ninguna ruta supera la capacidad Q.

    Parameters
    ----------
    routes : list[list[int]]
    instance : VRPInstance

    Returns
    -------
    (is_feasible, errors)
        is_feasible : bool
        errors : list[str] — mensajes de violación (vacío si es factible).
    """
    errors: list[str] = []
    n = instance.n_customers
    served: list[int] = []

    for r_idx, route in enumerate(routes):
        label = f"Ruta {r_idx + 1}"

        # 1. Validar índices — separar válidos de inválidos
        valid_nodes: list[int] = []
        for node in route:
            if not (1 <= node <= n):
                errors.append(
                    f"{label}: nodo {node} fuera del rango válido [1, {n}]."
                )
            else:
                served.append(node)
                valid_nodes.append(node)

        # 3. Capacidad — solo sobre nodos válidos para evitar IndexError
        load = float(sum(instance.demands[i] for i in valid_nodes))
        if load > instance.capacity:
            errors.append(
                f"{label}: carga {load:.2f} supera la capacidad Q={instance.capacity}."
            )

    # 2. Cobertura
    counts = Counter(served)
    missing = sorted(set(range(1, n + 1)) - counts.keys())
    if missing:
        errors.append(f"Clientes no atendidos: {missing}.")

    duplicates = sorted(node for node, cnt in counts.items() if cnt > 1)
    if duplicates:
        errors.append(f"Clientes visitados más de una vez: {duplicates}.")

    return len(errors) == 0, errors


# ---------------------------------------------------------------------------
# Clase VRPSolution
# ---------------------------------------------------------------------------

@dataclass
class VRPSolution:
    """Solución de una instancia CVRP.

    Convención: ruta [1, 4, 7] = recorrido 0 → 1 → 4 → 7 → 0.

    Parameters
    ----------
    instance : VRPInstance
        Instancia a la que pertenece esta solución.
    routes : list[list[int]]
        Lista de rutas (listas de índices de cliente, 1-based).
        Las rutas vacías se descartan automáticamente.
    method : str
        Nombre del algoritmo o método que generó la solución.
    metadata : dict
        Información libre: parámetros, tiempo de cómputo, versión, etc.
    history : list[dict]
        Historial de iteraciones (para metaheurísticas).
        Formato sugerido por entrada: {"iteration": int, "cost": float, ...}.
    """

    instance: VRPInstance
    routes: list[list[int]]
    method: str = "Desconocido"
    metadata: dict[str, Any] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.routes = [r for r in self.routes if r]

    # ------------------------------------------------------------------
    # Métricas de ruta
    # ------------------------------------------------------------------

    def route_load(self, route: list[int]) -> float:
        """Carga total de una ruta (delegación a función pura)."""
        return route_load(route, self.instance)

    def route_cost(self, route: list[int]) -> float:
        """Costo (distancia) de una ruta (delegación a función pura)."""
        return route_cost(route, self.instance)

    # ------------------------------------------------------------------
    # Métricas de solución
    # ------------------------------------------------------------------

    def total_cost(self) -> float:
        """Costo total de la solución (suma de distancias de todas las rutas)."""
        return round(solution_cost(self.routes, self.instance), 4)

    def n_vehicles(self) -> int:
        """Número de vehículos utilizados."""
        return len(self.routes)

    def capacity_utilizations(self) -> list[float]:
        """Porcentaje de utilización de capacidad Q por ruta, en el mismo orden."""
        return [
            round(route_load(r, self.instance) / self.instance.capacity * 100, 1)
            for r in self.routes
        ]

    def summary(self) -> dict[str, Any]:
        """Resumen compacto de la solución."""
        utils = self.capacity_utilizations()
        feasible, _ = check_feasibility(self.routes, self.instance)
        return {
            "Método": self.method,
            "Costo total": self.total_cost(),
            "Vehículos usados": self.n_vehicles(),
            "Utilización promedio (%)": round(sum(utils) / len(utils), 1) if utils else 0.0,
            "Factible": feasible,
        }

    # ------------------------------------------------------------------
    # Factibilidad
    # ------------------------------------------------------------------

    def is_feasible(self) -> bool:
        """True si la solución satisface todas las restricciones CVRP."""
        ok, _ = check_feasibility(self.routes, self.instance)
        return ok

    def feasibility_report(self) -> tuple[bool, list[str]]:
        """Informe detallado de factibilidad: (es_factible, lista_de_errores)."""
        return check_feasibility(self.routes, self.instance)

    # ------------------------------------------------------------------
    # Exportar
    # ------------------------------------------------------------------

    def to_routes_dataframe(self) -> pd.DataFrame:
        """Devuelve un DataFrame con una fila por ruta.

        Columnas
        --------
        Ruta, N clientes, Carga, Capacidad, Utilización (%), Costo, Secuencia
        """
        D = self.instance.distance_matrix()
        records = []
        for i, route in enumerate(self.routes):
            load = route_load(route, self.instance)
            cost = route_cost(route, self.instance, D)
            secuencia = "0 → " + " → ".join(map(str, route)) + " → 0"
            records.append({
                "Ruta": i + 1,
                "N clientes": len(route),
                "Carga": round(load, 2),
                "Capacidad": self.instance.capacity,
                "Utilización (%)": round(load / self.instance.capacity * 100, 1),
                "Costo": round(cost, 4),
                "Secuencia": secuencia,
            })
        return pd.DataFrame(records)

    def to_arcs_dataframe(self) -> pd.DataFrame:
        """Devuelve un DataFrame con una fila por arco recorrido.

        La ruta ``[1, 4, 7]`` genera los arcos
        ``(0,1), (1,4), (4,7), (7,0)``.

        Columnas
        --------
        Ruta, from_node, to_node, dist
        """
        D = self.instance.distance_matrix()
        records = []
        for r_idx, route in enumerate(self.routes):
            nodes = [0] + route + [0]
            for a, b in zip(nodes, nodes[1:]):
                records.append({
                    "Ruta": r_idx + 1,
                    "from_node": a,
                    "to_node": b,
                    "dist": round(float(D[a, b]), 6),
                })
        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Visualización
    # ------------------------------------------------------------------

    def plot_solution(
        self,
        title: Optional[str] = None,
        figsize: tuple[float, float] = (8, 6),
        show_demands: bool = True,
    ) -> plt.Figure:
        """Genera una figura Matplotlib con las rutas de la solución.

        Cada ruta tiene un color distinto. El depósito aparece como cuadrado
        negro. Los clientes se etiquetan con su índice (y demanda opcional).

        Parameters
        ----------
        title : str, optional
            Título del gráfico. Por defecto muestra método y costo.
        figsize : tuple[float, float]
        show_demands : bool
            Si True, añade la demanda junto al índice de cada cliente.

        Returns
        -------
        matplotlib.figure.Figure
        """
        inst = self.instance
        colors = list(mcolors.TABLEAU_COLORS.values())

        fig, ax = plt.subplots(figsize=figsize)

        # --- Rutas ---
        for idx, route in enumerate(self.routes):
            color = colors[idx % len(colors)]
            load = route_load(route, inst)
            nodes = [0] + route + [0]
            xs = [inst.coords[n, 0] for n in nodes]
            ys = [inst.coords[n, 1] for n in nodes]
            ax.plot(
                xs, ys, "-o",
                color=color, linewidth=1.8, markersize=5,
                label=f"Ruta {idx + 1}  (carga {load:.1f})",
                zorder=2,
            )

        # --- Depósito ---
        ax.plot(
            inst.coords[0, 0], inst.coords[0, 1],
            "ks", markersize=13, zorder=5, label="Depósito",
        )
        ax.annotate(
            "Depósito", xy=inst.coords[0],
            xytext=(5, 6), textcoords="offset points",
            fontsize=8, fontweight="bold",
        )

        # --- Clientes ---
        for i in range(1, inst.n_nodes):
            x, y = inst.coords[i]
            lbl = str(i) if not show_demands else f"{i}\n(d={inst.demands[i]:.0f})"
            ax.annotate(
                lbl, xy=(x, y),
                xytext=(4, 4), textcoords="offset points",
                fontsize=6.5, color="dimgray",
            )

        ax.set_title(
            title or f"{self.method}  |  Costo: {self.total_cost():.2f}  |  Q={inst.capacity}"
        )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.legend(loc="upper right", fontsize=8, framealpha=0.8)
        ax.grid(True, linestyle="--", alpha=0.35)
        fig.tight_layout()
        return fig


# ---------------------------------------------------------------------------
# Pruebas simples — instancia pequeña
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    from app.models.instances import generate_cvrp_instance

    print("=" * 55)
    print("Pruebas de solution.py")
    print("=" * 55)

    # Instancia de prueba: 6 clientes, Q=30
    inst = generate_cvrp_instance(n_customers=6, capacity=30, seed=0)
    print(f"\nInstancia: {inst.n_customers} clientes, Q={inst.capacity}")
    print(f"Demandas: {inst.demands[1:].tolist()}")

    # Rutas manuales (factibles por construcción)
    routes_ok = [[1, 2, 3], [4, 5, 6]]

    # ------------------------------------------------------------------
    # 1. route_load
    # ------------------------------------------------------------------
    load1 = route_load(routes_ok[0], inst)
    load2 = route_load(routes_ok[1], inst)
    print(f"\n[route_load]")
    print(f"  Ruta [1,2,3]: carga = {load1:.1f}")
    print(f"  Ruta [4,5,6]: carga = {load2:.1f}")
    assert load1 == inst.demands[1] + inst.demands[2] + inst.demands[3]
    assert load2 == inst.demands[4] + inst.demands[5] + inst.demands[6]
    print("  OK")

    # ------------------------------------------------------------------
    # 2. route_cost
    # ------------------------------------------------------------------
    D = inst.distance_matrix()
    cost1 = route_cost(routes_ok[0], inst, D)
    cost1_direct = D[0, 1] + D[1, 2] + D[2, 3] + D[3, 0]
    print(f"\n[route_cost]")
    print(f"  Ruta [1,2,3]: costo = {cost1:.4f}")
    assert abs(cost1 - cost1_direct) < 1e-9, "route_cost incorrecto"
    assert route_cost([], inst) == 0.0, "ruta vacía debe devolver 0"
    print("  OK")

    # ------------------------------------------------------------------
    # 3. solution_cost
    # ------------------------------------------------------------------
    total = solution_cost(routes_ok, inst)
    manual = route_cost(routes_ok[0], inst, D) + route_cost(routes_ok[1], inst, D)
    print(f"\n[solution_cost]")
    print(f"  Total: {total:.4f}")
    assert abs(total - manual) < 1e-9
    print("  OK")

    # ------------------------------------------------------------------
    # 4. check_feasibility — caso factible
    # ------------------------------------------------------------------
    ok, errors = check_feasibility(routes_ok, inst)
    print(f"\n[check_feasibility — factible]")
    print(f"  Factible: {ok}, errores: {errors}")
    assert ok and errors == []
    print("  OK")

    # ------------------------------------------------------------------
    # 5. check_feasibility — cliente duplicado
    # ------------------------------------------------------------------
    routes_dup = [[1, 2, 3], [3, 4, 5, 6]]
    ok2, err2 = check_feasibility(routes_dup, inst)
    print(f"\n[check_feasibility — duplicado]")
    print(f"  Factible: {ok2}, errores: {err2}")
    assert not ok2
    assert any("más de una vez" in e for e in err2)
    print("  OK")

    # ------------------------------------------------------------------
    # 6. check_feasibility — cliente faltante
    # ------------------------------------------------------------------
    routes_miss = [[1, 2, 3], [4, 5]]
    ok3, err3 = check_feasibility(routes_miss, inst)
    print(f"\n[check_feasibility — faltante]")
    print(f"  Factible: {ok3}, errores: {err3}")
    assert not ok3
    assert any("no atendidos" in e for e in err3)
    print("  OK")

    # ------------------------------------------------------------------
    # 7. check_feasibility — capacidad superada
    # ------------------------------------------------------------------
    # Forzar demanda total de clientes 1..6 en una sola ruta
    routes_over = [[1, 2, 3, 4, 5, 6]]
    ok4, err4 = check_feasibility(routes_over, inst)
    print(f"\n[check_feasibility — capacidad]")
    print(f"  Factible: {ok4}, errores: {err4}")
    # Puede ser factible si la demanda total cabe en Q;
    # si no, verifica el mensaje
    total_dem = sum(inst.demands[i] for i in range(1, 7))
    if total_dem > inst.capacity:
        assert not ok4
        assert any("capacidad" in e for e in err4)
        print("  Capacidad superada detectada — OK")
    else:
        assert ok4
        print("  Caben todos en una ruta — OK")

    # ------------------------------------------------------------------
    # 8. check_feasibility — nodo inválido
    # ------------------------------------------------------------------
    routes_bad = [[1, 2, 99], [3, 4, 5, 6]]
    ok5, err5 = check_feasibility(routes_bad, inst)
    print(f"\n[check_feasibility — nodo inválido]")
    print(f"  Factible: {ok5}, errores: {err5}")
    assert not ok5
    assert any("99" in e for e in err5)
    print("  OK")

    # ------------------------------------------------------------------
    # 9. VRPSolution — to_routes_dataframe
    # ------------------------------------------------------------------
    sol = VRPSolution(
        instance=inst,
        routes=routes_ok,
        method="Manual",
        metadata={"descripcion": "rutas de prueba"},
        history=[{"iteration": 0, "cost": total}],
    )
    df = sol.to_routes_dataframe()
    print(f"\n[VRPSolution.to_routes_dataframe]")
    # Reemplaza flechas Unicode para compatibilidad con consolas cp1252
    safe = df.to_string(index=False).encode("ascii", "replace").decode("ascii")
    print(safe)
    assert len(df) == 2
    assert list(df["Ruta"]) == [1, 2]
    assert "Secuencia" in df.columns
    print("  OK")

    # ------------------------------------------------------------------
    # 10. VRPSolution — summary e is_feasible
    # ------------------------------------------------------------------
    print(f"\n[VRPSolution.summary]")
    print(sol.summary())
    assert sol.is_feasible()
    assert sol.total_cost() == round(total, 4)
    print("  OK")

    # ------------------------------------------------------------------
    # 11. VRPSolution — plot_solution (guarda a archivo)
    # ------------------------------------------------------------------
    fig = sol.plot_solution(show_demands=True)
    out = "prueba_solution.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    assert os.path.exists(out)
    print(f"\n[plot_solution] Figura guardada en '{out}'  OK")

    print("\n" + "=" * 55)
    print("Todas las pruebas pasaron.")
    print("=" * 55)
