"""registry.py — Registro centralizado de métodos de solución CVRP.

Contrato uniforme para todos los métodos:
    solver = cls(**init_params)
    sol    = solver.solve(instance, seed=seed)   →  solution.VRPSolution

METHOD_REGISTRY  : {nombre: clase}
PARAMS_SPEC      : {nombre: [dict_param, ...]}  — describe widgets de la UI.
"""

from __future__ import annotations

import time
from typing import Any, Optional

from app.algorithms.heuristics.nearest_neighbor import NearestNeighbor
from app.models.instances import VRPInstance
from app.models.solution import VRPSolution


# ---------------------------------------------------------------------------
# Adaptadores para heurísticas legacy
# ---------------------------------------------------------------------------

class _ClarkeWrightSolver:
    """Envuelve clarke_wright.solve en el contrato solve(instance, seed)."""

    def solve(self, instance: VRPInstance, seed: Optional[int] = None) -> VRPSolution:
        from app.algorithms.heuristics import clarke_wright
        t = time.perf_counter()
        legacy = clarke_wright.solve(instance)
        return VRPSolution(
            instance=instance,
            routes=legacy.routes,
            method="Clarke & Wright Savings",
            metadata={"computation_time_s": round(time.perf_counter() - t, 6)},
        )


class _SweepSolver:
    """Envuelve sweep.solve en el contrato solve(instance, seed)."""

    def solve(self, instance: VRPInstance, seed: Optional[int] = None) -> VRPSolution:
        from app.algorithms.heuristics import sweep
        t = time.perf_counter()
        legacy = sweep.solve(instance)
        return VRPSolution(
            instance=instance,
            routes=legacy.routes,
            method="Sweep Algorithm",
            metadata={"computation_time_s": round(time.perf_counter() - t, 6)},
        )


# ---------------------------------------------------------------------------
# Especificación de parámetros para la UI
#
# Claves reconocidas por app.py:
#   key      → nombre del kwarg del constructor
#   widget   → "selectbox" | "number_input" | "checkbox"
#   label    → texto visible en la UI
#   options  → lista de opciones (solo para selectbox)
#   default  → valor por defecto
#   help     → tooltip
#   min_value, max_value, step → opcionales para number_input
# ---------------------------------------------------------------------------

PARAMS_SPEC: dict[str, list[dict[str, Any]]] = {
    "Nearest Neighbor": [
        {
            "key": "start_strategy",
            "widget": "selectbox",
            "label": "Estrategia de inicio",
            "options": ["depot", "random", "farthest"],
            "default": "depot",
            "help": (
                "depot: greedy puro desde el depósito.  "
                "random: primer cliente al azar.  "
                "farthest: el cliente más lejano al depósito."
            ),
        },
        {
            "key": "tie_break",
            "widget": "selectbox",
            "label": "Regla de desempate",
            "options": ["first", "random", "lowest_demand"],
            "default": "first",
            "help": (
                "first: menor índice de nodo.  "
                "random: elección aleatoria.  "
                "lowest_demand: menor demanda entre empatados."
            ),
        },
        {
            "key": "seed",
            "widget": "number_input",
            "label": "Semilla aleatoria",
            "default": 42,
            "min_value": 0,
            "max_value": 99999,
            "step": 1,
            "help": "Semilla para reproducibilidad (activa con start=random o tie=random).",
        },
    ],
    "Clarke & Wright Savings": [],
    "Sweep Algorithm": [],
}


# ---------------------------------------------------------------------------
# Descripciones pedagógicas por método (Markdown)
# ---------------------------------------------------------------------------

METHOD_DESCRIPTIONS: dict[str, str] = {
    "Nearest Neighbor": """\
**Vecino Más Cercano** — heurística constructiva greedy.

Desde el depósito (o el primer cliente según la estrategia), elige siempre
el cliente **no visitado más cercano** que quepa en la capacidad restante.
Cuando no hay más candidatos, cierra la ruta y abre una nueva.

| Propiedad | Valor |
|-----------|-------|
| Complejidad | O(n²) |
| Calidad | Buena en instancias pequeñas; subóptima en grandes |
| Reproducible | Sí, con semilla fija |
| Parámetros clave | Estrategia de inicio, regla de desempate |
""",
    "Clarke & Wright Savings": """\
**Clarke & Wright Savings** — heurística de fusión de rutas (1964).

Parte de *n* rutas individuales (depósito → cliente → depósito) y las
**fusiona en orden decreciente de ahorro**:

> s(i, j) = d(0, i) + d(0, j) − d(i, j)

Una fusión se acepta si la carga combinada no supera Q.

| Propiedad | Valor |
|-----------|-------|
| Complejidad | O(n² log n) |
| Calidad | Generalmente mejor que Nearest Neighbor |
| Reproducible | Siempre (determinista) |
| Parámetros clave | Ninguno configurable |
""",
    "Sweep Algorithm": """\
**Sweep Algorithm** — heurística de barrido angular (Gillett & Miller, 1974).

Ordena los clientes por **ángulo polar** respecto al depósito y los asigna
secuencialmente a rutas, abriendo una nueva ruta al superar la capacidad Q.

| Propiedad | Valor |
|-----------|-------|
| Complejidad | O(n log n) |
| Calidad | Buena cuando la distribución geográfica es radial |
| Reproducible | Siempre (determinista) |
| Parámetros clave | Ninguno configurable |
""",
}


# ---------------------------------------------------------------------------
# Registro principal
# ---------------------------------------------------------------------------

METHOD_REGISTRY: dict[str, type] = {
    "Nearest Neighbor": NearestNeighbor,
    "Clarke & Wright Savings": _ClarkeWrightSolver,
    "Sweep Algorithm": _SweepSolver,
}
