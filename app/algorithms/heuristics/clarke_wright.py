import time

from app.models import VRPInstance, VRPSolution


def solve(instance: VRPInstance) -> VRPSolution:
    """Heurística Clarke & Wright Savings.

    Calcula savings s(i,j) = d(0,i) + d(0,j) - d(i,j) y fusiona rutas en
    orden decreciente de saving mientras no se supere la capacidad Q.
    """
    start = time.time()
    matrix = instance.distance_matrix()
    n = instance.n_customers

    # Calcular savings entre todos los pares de clientes
    savings: list[tuple[float, int, int]] = []
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            s = matrix[0][i] + matrix[0][j] - matrix[i][j]
            savings.append((s, i, j))
    savings.sort(reverse=True)

    # Inicializar una ruta individual por cliente
    routes: list[list[int]] = [[i] for i in range(1, n + 1)]
    route_of: dict[int, int] = {i: i - 1 for i in range(1, n + 1)}

    def route_demand(r_idx: int) -> float:
        return sum(instance.demands[c] for c in routes[r_idx])

    for _, i, j in savings:
        ri = route_of.get(i)
        rj = route_of.get(j)
        if ri is None or rj is None or ri == rj:
            continue
        if not routes[ri] or not routes[rj]:
            continue
        # i debe ser el último de ri y j el primero de rj
        if routes[ri][-1] != i or routes[rj][0] != j:
            continue
        if route_demand(ri) + route_demand(rj) > instance.capacity:
            continue
        # Fusionar rj al final de ri
        for c in routes[rj]:
            route_of[c] = ri
        routes[ri].extend(routes[rj])
        routes[rj] = []

    active_routes = [r for r in routes if r]

    return VRPSolution(
        instance=instance,
        routes=active_routes,
        algorithm_name="Clarke & Wright Savings",
        computation_time=time.time() - start,
    )
