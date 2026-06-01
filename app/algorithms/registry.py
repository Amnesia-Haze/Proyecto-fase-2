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
# Registro principal
# ---------------------------------------------------------------------------

METHOD_REGISTRY: dict[str, type] = {
    "Nearest Neighbor": NearestNeighbor,
    "Clarke & Wright Savings": _ClarkeWrightSolver,
    "Sweep Algorithm": _SweepSolver,
}
