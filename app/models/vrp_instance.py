from __future__ import annotations

from dataclasses import dataclass

from app.models.instances import VRPInstance


@dataclass
class VRPSolution:
    """Solución de una instancia CVRP.

    routes: lista de rutas; cada ruta es una lista de índices de nodo (1-based,
    igual que en coords/demands). El depósito (nodo 0) se añade implícitamente
    al inicio y fin de cada ruta al calcular distancias.
    """

    instance: VRPInstance
    routes: list[list[int]]
    algorithm_name: str
    computation_time: float = 0.0

    def total_distance(self) -> float:
        """Distancia total recorrida por todos los vehículos."""
        matrix = self.instance.distance_matrix()
        total = 0.0
        for route in self.routes:
            if not route:
                continue
            total += matrix[0][route[0]]
            for i in range(len(route) - 1):
                total += matrix[route[i]][route[i + 1]]
            total += matrix[route[-1]][0]
        return round(total, 4)

    def vehicles_used(self) -> int:
        """Número de vehículos con al menos un cliente."""
        return sum(1 for r in self.routes if r)

    def capacity_utilization(self) -> list[float]:
        """Porcentaje de utilización de capacidad por ruta activa."""
        utilizations = []
        for route in self.routes:
            if not route:
                continue
            demand = sum(self.instance.demands[i] for i in route)
            utilizations.append(round(demand / self.instance.capacity * 100, 1))
        return utilizations

    def summary(self) -> dict:
        utils = self.capacity_utilization()
        return {
            "Algoritmo": self.algorithm_name,
            "Distancia total": self.total_distance(),
            "Vehículos utilizados": self.vehicles_used(),
            "Tiempo (s)": round(self.computation_time, 4),
            "Utilización promedio (%)": round(sum(utils) / len(utils), 1) if utils else 0,
        }
