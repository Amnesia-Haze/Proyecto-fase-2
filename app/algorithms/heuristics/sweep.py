import math
import time

from app.models import VRPInstance, VRPSolution


def solve(instance: VRPInstance) -> VRPSolution:
    """Heurística Sweep: agrupa clientes por ángulo polar respecto al depósito.

    Ordena los clientes por ángulo (sentido antihorario desde el este) y los
    asigna secuencialmente a una ruta; abre nueva ruta al superar la capacidad.
    """
    start = time.time()

    depot_x, depot_y = instance.coords[0]

    customers_by_angle: list[tuple[float, int]] = []
    for i in range(1, instance.n_nodes):
        angle = math.atan2(
            instance.coords[i, 1] - depot_y,
            instance.coords[i, 0] - depot_x,
        )
        customers_by_angle.append((angle, i))
    customers_by_angle.sort(key=lambda t: t[0])

    routes: list[list[int]] = []
    current_route: list[int] = []
    current_demand = 0.0

    for _, i in customers_by_angle:
        d = instance.demands[i]
        if current_demand + d > instance.capacity:
            if current_route:
                routes.append(current_route)
            current_route = [i]
            current_demand = d
        else:
            current_route.append(i)
            current_demand += d

    if current_route:
        routes.append(current_route)

    return VRPSolution(
        instance=instance,
        routes=routes,
        algorithm_name="Sweep Algorithm",
        computation_time=time.time() - start,
    )
