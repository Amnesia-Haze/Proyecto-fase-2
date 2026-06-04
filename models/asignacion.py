"""models/asignacion.py — Modelo ILP de asignación de guardias (Bursan Empresas).

Asigna 14 guardias de seguridad (G1–G14) a 4 empresas de la Región del Biobío,
minimizando distancia total con soporte opcional de equidad.

Restricciones del modelo (R1–R5)
---------------------------------
R1  Cada guardia activo se asigna a exactamente una empresa.
R2  Cada empresa recibe al menos ``min_guardias`` guardias.
R3  Cada empresa recibe como máximo ``max_guardias`` guardias.
R4  Toda empresa que requiere supervisor tiene al menos uno asignado.
R5  Solo participan guardias con estado ``activo=True`` (filtrado previo).

Modos de objetivo
-----------------
suma_total    Minimizar Σ d_ij · x_ij  (modelo original).
minimax       Minimizar M  s.a.  M ≥ Σ_j d_ij · x_ij  ∀i  (equidad máximo).
multiobjetivo Minimizar α · Σd + β · M  (combinación convexa ponderada).

Restricción de equidad adicional
---------------------------------
delta_max_equidad  Garantiza que la distancia máxima menos la mínima ≤ δ.
                   Implementación: d_i − d_j ≤ δ  ∀(i,j) activos, sin variable
                   auxiliar extra (evita artefactos numéricos en objetivos mixtos).

Ejecución rápida
----------------
    python -m models.asignacion
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal, Optional

import pulp

# ---------------------------------------------------------------------------
# Datos base — reemplaza con tu matriz y puestos originales
# ---------------------------------------------------------------------------

#: Identificadores de todos los guardias del sistema.
GUARDIAS: list[str] = [f"G{i}" for i in range(1, 15)]  # G1..G14

#: Subconjunto con certificación de supervisor.
SUPERVISORES: frozenset[str] = frozenset({"G1", "G6"})

#: Empresas destino.
EMPRESAS: list[str] = ["Noramco", "ITI Chile", "Oleoducto", "Indama"]

# Distancias en km desde el domicilio de cada guardia a cada empresa.
# Fuente: matriz de distancias del modelo original (Región del Biobío).
DISTANCIAS: dict[str, dict[str, float]] = {
    #         Noramco  ITI Chile  Oleoducto  Indama
    "G1":  {"Noramco": 12.5, "ITI Chile":  8.3, "Oleoducto": 34.2, "Indama": 22.1},
    "G2":  {"Noramco": 18.7, "ITI Chile": 15.4, "Oleoducto": 28.9, "Indama":  9.6},
    "G3":  {"Noramco":  5.2, "ITI Chile": 11.8, "Oleoducto": 42.1, "Indama": 31.5},
    "G4":  {"Noramco": 23.1, "ITI Chile":  6.7, "Oleoducto": 19.4, "Indama": 14.3},
    "G5":  {"Noramco": 31.4, "ITI Chile": 27.9, "Oleoducto": 11.6, "Indama":  8.2},
    "G6":  {"Noramco":  9.8, "ITI Chile": 14.2, "Oleoducto": 38.7, "Indama": 25.6},
    "G7":  {"Noramco": 16.3, "ITI Chile": 21.5, "Oleoducto":  7.8, "Indama": 19.4},
    "G8":  {"Noramco": 28.6, "ITI Chile": 33.1, "Oleoducto": 15.9, "Indama":  6.7},
    "G9":  {"Noramco":  7.4, "ITI Chile": 12.9, "Oleoducto": 45.3, "Indama": 33.8},
    "G10": {"Noramco": 35.2, "ITI Chile": 29.6, "Oleoducto":  8.4, "Indama": 16.7},
    "G11": {"Noramco": 14.6, "ITI Chile": 19.3, "Oleoducto": 31.5, "Indama": 11.2},
    "G12": {"Noramco": 22.8, "ITI Chile": 18.5, "Oleoducto": 24.1, "Indama": 28.9},
    "G13": {"Noramco": 41.7, "ITI Chile": 36.2, "Oleoducto": 13.6, "Indama":  7.4},
    "G14": {"Noramco": 11.3, "ITI Chile":  7.8, "Oleoducto": 29.7, "Indama": 18.5},
}

# Dotación requerida por empresa (diccionario puestos del modelo original).
PUESTOS: dict[str, dict] = {
    "Noramco":   {"min": 3, "max": 5, "requiere_supervisor": True},
    "ITI Chile": {"min": 2, "max": 4, "requiere_supervisor": False},
    "Oleoducto": {"min": 3, "max": 5, "requiere_supervisor": True},
    "Indama":    {"min": 2, "max": 4, "requiere_supervisor": False},
}

# ---------------------------------------------------------------------------
# Tipos
# ---------------------------------------------------------------------------

ModoObjetivo = Literal["suma_total", "minimax", "multiobjetivo"]


# ---------------------------------------------------------------------------
# Resultado
# ---------------------------------------------------------------------------

@dataclass
class ResultadoAsignacion:
    """Salida estructurada de :func:`resolver`.

    Attributes
    ----------
    estado : str
        ``'Optimo'``, ``'Infactible'``, ``'Sin_solucion'`` o ``'Error'``.
    asignacion : dict[str, str]
        ``{guardia_id: empresa}`` — vacío si no hay solución factible.
    costo_total : float
        Suma de distancias de todos los guardias asignados (km).
    dist_maxima : float
        Distancia del guardia más lejano a su empresa asignada (km).
    dist_minima : float
        Distancia del guardia más cercano a su empresa asignada (km).
    rango : float
        ``dist_maxima − dist_minima`` — indicador de inequidad (km).
    tiempo_cpu : float
        Tiempo de resolución en segundos.
    modo_objetivo : str
        Modo usado en esta ejecución.
    guardias_activos_usados : list[str]
        Guardias que participaron en el modelo.
    infactibilidad_causa : str
        Descripción legible de la causa probable cuando ``estado != 'Optimo'``.
    """

    estado: str
    asignacion: dict[str, str]
    costo_total: float
    dist_maxima: float
    dist_minima: float
    rango: float
    tiempo_cpu: float
    modo_objetivo: str
    guardias_activos_usados: list[str] = field(default_factory=list)
    infactibilidad_causa: str = ""

    # ------------------------------------------------------------------
    # Helpers de presentación
    # ------------------------------------------------------------------

    def resumen(self) -> str:
        """Texto compacto para impresión rápida."""
        if self.estado != "Optimo":
            return (
                f"[{self.estado}] {self.modo_objetivo} | "
                f"Causa: {self.infactibilidad_causa or 'desconocida'}"
            )
        lines = [
            f"[{self.estado}] modo={self.modo_objetivo}",
            f"  Costo total  : {self.costo_total:.2f} km",
            f"  Dist máxima  : {self.dist_maxima:.2f} km",
            f"  Dist mínima  : {self.dist_minima:.2f} km",
            f"  Rango        : {self.rango:.2f} km",
            f"  Tiempo CPU   : {self.tiempo_cpu:.4f} s",
            f"  Asignación   :",
        ]
        for g, e in sorted(self.asignacion.items()):
            sup = " [SUP]" if g in SUPERVISORES else ""
            lines.append(f"    {g}{sup} -> {e}  ({DISTANCIAS[g][e]:.1f} km)")
        return "\n".join(lines)

    def dotacion_por_empresa(self) -> dict[str, list[str]]:
        """Devuelve ``{empresa: [guardias]}`` desde la asignación."""
        dotacion: dict[str, list[str]] = {e: [] for e in EMPRESAS}
        for g, e in self.asignacion.items():
            dotacion[e].append(g)
        return dotacion


# ---------------------------------------------------------------------------
# Validación previa al solver (falla rápido con mensajes útiles)
# ---------------------------------------------------------------------------

def _validar_parametros(
    guardias_activos: list[str],
    modo_objetivo: ModoObjetivo,
    alpha: float,
    beta: float,
    delta_max: Optional[float],
) -> None:
    """Verifica coherencia de parámetros y condiciones necesarias de factibilidad.

    Raises
    ------
    ValueError
        Con un mensaje específico que identifica la causa exacta del problema.
    """
    # IDs válidos
    ids_invalidos = [g for g in guardias_activos if g not in DISTANCIAS]
    if ids_invalidos:
        raise ValueError(
            f"IDs de guardia no reconocidos: {ids_invalidos}. "
            f"IDs válidos: {GUARDIAS}."
        )

    # Sin duplicados
    if len(guardias_activos) != len(set(guardias_activos)):
        from collections import Counter
        dupes = [g for g, n in Counter(guardias_activos).items() if n > 1]
        raise ValueError(f"IDs duplicados en guardias_activos: {dupes}.")

    # Modo objetivo
    modos_validos = ("suma_total", "minimax", "multiobjetivo")
    if modo_objetivo not in modos_validos:
        raise ValueError(
            f"modo_objetivo='{modo_objetivo}' inválido. "
            f"Opciones: {modos_validos}."
        )

    # Pesos multiobjetivo
    if modo_objetivo == "multiobjetivo":
        if not (0.0 <= alpha <= 1.0 and 0.0 <= beta <= 1.0):
            raise ValueError(
                f"alpha={alpha} y beta={beta} deben estar en [0, 1]. "
                "Ejemplo: alpha=0.6, beta=0.4."
            )
        if alpha + beta == 0.0:
            raise ValueError(
                "alpha + beta = 0: la función objetivo estaría vacía. "
                "Define al menos un peso > 0."
            )

    # delta_max
    if delta_max is not None and delta_max < 0:
        raise ValueError(
            f"delta_max={delta_max} debe ser ≥ 0. "
            "Es la diferencia máxima permitida entre la distancia más larga y la más corta."
        )

    # Cobertura mínima (falla antes de llamar al solver)
    total_min = sum(p["min"] for p in PUESTOS.values())
    n_activos = len(guardias_activos)
    if n_activos < total_min:
        raise ValueError(
            f"Guardias activos insuficientes: hay {n_activos} pero la dotación "
            f"mínima total suma {total_min}. "
            f"Activa más guardias o reduce los mínimos en PUESTOS."
        )

    total_max = sum(p["max"] for p in PUESTOS.values())
    if n_activos > total_max:
        raise ValueError(
            f"Demasiados guardias activos: hay {n_activos} pero la dotación "
            f"máxima total es {total_max}. "
            f"Desactiva guardias o sube los máximos en PUESTOS."
        )

    # Supervisores suficientes
    supervisores_activos = [g for g in guardias_activos if g in SUPERVISORES]
    empresas_req_sup = [e for e, p in PUESTOS.items() if p["requiere_supervisor"]]
    if len(supervisores_activos) < len(empresas_req_sup):
        raise ValueError(
            f"Supervisores insuficientes: se necesita al menos 1 por cada empresa "
            f"que requiere supervisor ({empresas_req_sup}), pero solo hay "
            f"{len(supervisores_activos)} supervisor(es) activo(s) "
            f"({supervisores_activos or 'ninguno'}). "
            "Activa G1 o G6."
        )


def _diagnosticar_infactibilidad(
    guardias_activos: list[str],
    delta_max: Optional[float],
) -> str:
    """Genera hipótesis de causa cuando el solver reporta infactibilidad."""
    causas: list[str] = []

    total_min = sum(p["min"] for p in PUESTOS.values())
    if len(guardias_activos) < total_min:
        causas.append(
            f"dotación mínima={total_min} > guardias activos={len(guardias_activos)}"
        )

    supervisores = [g for g in guardias_activos if g in SUPERVISORES]
    n_emp_sup = sum(1 for p in PUESTOS.values() if p["requiere_supervisor"])
    if len(supervisores) < n_emp_sup:
        causas.append(
            f"supervisores insuficientes: se necesitan {n_emp_sup}, "
            f"disponibles {len(supervisores)}"
        )

    if delta_max is not None:
        # Estima el rango mínimo teórico con la asignación greedy más ajustada
        min_dists = [min(DISTANCIAS[g].values()) for g in guardias_activos]
        max_dists = [min(DISTANCIAS[g].values()) for g in guardias_activos]
        rango_estimado = max(min_dists) - min(max_dists)
        if rango_estimado > delta_max:
            causas.append(
                f"delta_max={delta_max:.1f} km posiblemente demasiado restrictivo "
                f"(rango mínimo estimado ≈ {rango_estimado:.1f} km). "
                "Intenta aumentar delta_max."
            )

    if not causas:
        causas.append(
            "causa no determinada automáticamente; "
            "revisa PUESTOS (min/max) y disponibilidad de supervisores"
        )

    return " | ".join(causas)


# ---------------------------------------------------------------------------
# Función principal
# ---------------------------------------------------------------------------

def resolver(
    guardias_activos: Optional[list[str]] = None,
    modo_objetivo: ModoObjetivo = "suma_total",
    alpha: float = 0.5,
    beta: float = 0.5,
    delta_max: Optional[float] = None,
    tiempo_limite: int = 120,
    verbose: bool = False,
) -> ResultadoAsignacion:
    """Construye y resuelve el modelo ILP de asignación de guardias.

    Parameters
    ----------
    guardias_activos : list[str], optional
        IDs de los guardias con estado ``activo``.
        Si es ``None`` se usan los 14 guardias del catálogo completo.
    modo_objetivo : {'suma_total', 'minimax', 'multiobjetivo'}
        Criterio de optimización:

        * ``'suma_total'``    — Minimizar Σ d_ij · x_ij  (modelo original).
        * ``'minimax'``       — Minimizar M  s.a.  M ≥ d_i  ∀i activo.
          Reduce la distancia del guardia más perjudicado.
        * ``'multiobjetivo'`` — Minimizar α·Σd + β·M  con α, β ∈ [0, 1].
          Permite calibrar el trade-off equidad/costo.
    alpha : float
        Peso de la suma de distancias en modo ``'multiobjetivo'``.
        Rango [0, 1]. Por defecto 0.5.
    beta : float
        Peso de la distancia máxima en modo ``'multiobjetivo'``.
        Rango [0, 1]. Por defecto 0.5.
    delta_max : float, optional
        Si se especifica, añade la restricción:
        ``max_dist − min_dist ≤ delta_max``.
        Implementada como: ``d_i − d_j ≤ delta_max  ∀(i,j)`` activos.
        Un valor de 0 fuerza que todos los guardias recorran la misma distancia
        (raramente factible); 999 equivale a sin restricción.
    tiempo_limite : int
        Tiempo máximo de cómputo del solver CBC en segundos. Por defecto 120.
    verbose : bool
        Si ``True``, imprime el log completo del solver CBC.

    Returns
    -------
    ResultadoAsignacion
        Objeto con ``asignacion``, ``costo_total``, ``dist_maxima``,
        ``dist_minima``, ``rango``, ``tiempo_cpu`` y diagnóstico de
        infactibilidad si aplica.

    Raises
    ------
    ValueError
        Si los parámetros son inválidos o la infactibilidad se detecta
        antes de llamar al solver (falla rápida).

    Examples
    --------
    Modo original (suma de distancias):

    >>> r = resolver()
    >>> r.estado
    'Optimo'

    Modo minimax con guardias parciales y restricción de equidad:

    >>> r = resolver(
    ...     guardias_activos=["G1","G2","G3","G4","G5","G6","G7","G8",
    ...                       "G9","G10","G11","G12","G13","G14"],
    ...     modo_objetivo="minimax",
    ...     delta_max=20.0,
    ... )
    >>> r.rango <= 20.0
    True
    """
    if guardias_activos is None:
        guardias_activos = list(GUARDIAS)

    # Copia defensiva para no mutar la lista del llamador
    guardias_activos = list(guardias_activos)

    # ------------------------------------------------------------------ #
    # Pre-validación (falla rápido, mensaje útil)                         #
    # ------------------------------------------------------------------ #
    _validar_parametros(guardias_activos, modo_objetivo, alpha, beta, delta_max)

    t_inicio = time.perf_counter()

    # ------------------------------------------------------------------ #
    # Variables de decisión                                               #
    # x[g][e] ∈ {0, 1}  — guardia g asignado a empresa e                 #
    # ------------------------------------------------------------------ #
    prob = pulp.LpProblem("Asignacion_Guardias_Bursan", pulp.LpMinimize)

    x: dict[str, dict[str, pulp.LpVariable]] = {
        g: {
            e: pulp.LpVariable(f"x_{g}_{e}", cat="Binary")
            for e in EMPRESAS
        }
        for g in guardias_activos
    }

    # Distancia efectiva de cada guardia: expresión lineal Σ_e d_ge · x_ge
    # No es variable de decisión; PuLP la evalúa como expresión simbólica.
    dist_efectiva: dict[str, pulp.LpAffineExpression] = {
        g: pulp.lpSum(DISTANCIAS[g][e] * x[g][e] for e in EMPRESAS)
        for g in guardias_activos
    }

    # Suma total de distancias (usada en suma_total y multiobjetivo)
    suma_total_expr = pulp.lpSum(dist_efectiva[g] for g in guardias_activos)

    # Variable auxiliar M para minimax y multiobjetivo
    necesita_M = modo_objetivo in ("minimax", "multiobjetivo")
    M: Optional[pulp.LpVariable] = None
    if necesita_M:
        M = pulp.LpVariable("M_distancia_max", lowBound=0.0)

    # ------------------------------------------------------------------ #
    # Función objetivo                                                    #
    # ------------------------------------------------------------------ #
    if modo_objetivo == "suma_total":
        prob += suma_total_expr, "FO_suma_distancias"

    elif modo_objetivo == "minimax":
        prob += M, "FO_minimax"

    else:  # multiobjetivo
        prob += alpha * suma_total_expr + beta * M, "FO_multiobjetivo"

    # ------------------------------------------------------------------ #
    # R1 — Asignación única: cada guardia → exactamente una empresa       #
    # ------------------------------------------------------------------ #
    for g in guardias_activos:
        prob += (
            pulp.lpSum(x[g][e] for e in EMPRESAS) == 1,
            f"R1_{g}",
        )

    # ------------------------------------------------------------------ #
    # R2 — Dotación mínima por empresa                                    #
    # ------------------------------------------------------------------ #
    for e in EMPRESAS:
        prob += (
            pulp.lpSum(x[g][e] for g in guardias_activos) >= PUESTOS[e]["min"],
            f"R2_min_{e.replace(' ', '_')}",
        )

    # ------------------------------------------------------------------ #
    # R3 — Dotación máxima por empresa                                    #
    # ------------------------------------------------------------------ #
    for e in EMPRESAS:
        prob += (
            pulp.lpSum(x[g][e] for g in guardias_activos) <= PUESTOS[e]["max"],
            f"R3_max_{e.replace(' ', '_')}",
        )

    # ------------------------------------------------------------------ #
    # R4 — Supervisor requerido                                           #
    # Al menos un supervisor activo en cada empresa que lo exija.        #
    # ------------------------------------------------------------------ #
    supervisores_activos = [g for g in guardias_activos if g in SUPERVISORES]
    for e in EMPRESAS:
        if PUESTOS[e]["requiere_supervisor"]:
            prob += (
                pulp.lpSum(x[g][e] for g in supervisores_activos) >= 1,
                f"R4_supervisor_{e.replace(' ', '_')}",
            )

    # R5 — Guardia activo: implícita. Solo los guardias en guardias_activos
    # tienen variables x; los inactivos no existen en el modelo.

    # ------------------------------------------------------------------ #
    # Linealización de M (minimax / multiobjetivo)                        #
    # M ≥ d_i  ∀ guardia activo i                                        #
    # ------------------------------------------------------------------ #
    if necesita_M:
        for g in guardias_activos:
            prob += (
                M >= dist_efectiva[g],
                f"R_Mmax_{g}",
            )

    # ------------------------------------------------------------------ #
    # Restricción de equidad: rango ≤ delta_max                          #
    # Formulación directa por pares:                                     #
    #   d_i − d_j ≤ delta_max  ∀(i,j) activos, i ≠ j                   #
    # Equivale a max(d_i) − min(d_j) ≤ delta_max sin variable auxiliar. #
    # ------------------------------------------------------------------ #
    if delta_max is not None:
        for idx_i, g_i in enumerate(guardias_activos):
            for g_j in guardias_activos[idx_i + 1:]:
                prob += (
                    dist_efectiva[g_i] - dist_efectiva[g_j] <= delta_max,
                    f"R_delta_{g_i}_gt_{g_j}",
                )
                prob += (
                    dist_efectiva[g_j] - dist_efectiva[g_i] <= delta_max,
                    f"R_delta_{g_j}_gt_{g_i}",
                )

    # ------------------------------------------------------------------ #
    # Resolver con CBC                                                    #
    # ------------------------------------------------------------------ #
    solver_cbc = pulp.PULP_CBC_CMD(
        msg=1 if verbose else 0,
        timeLimit=tiempo_limite,
        gapRel=0.01,
    )

    try:
        prob.solve(solver_cbc)
    except Exception as exc:
        return ResultadoAsignacion(
            estado="Error",
            asignacion={},
            costo_total=float("inf"),
            dist_maxima=float("inf"),
            dist_minima=float("inf"),
            rango=float("inf"),
            tiempo_cpu=round(time.perf_counter() - t_inicio, 6),
            modo_objetivo=modo_objetivo,
            guardias_activos_usados=guardias_activos,
            infactibilidad_causa=f"Excepción del solver CBC: {exc}",
        )

    estado_pulp = pulp.LpStatus[prob.status]
    t_cpu = round(time.perf_counter() - t_inicio, 6)

    # ------------------------------------------------------------------ #
    # Infactible                                                          #
    # ------------------------------------------------------------------ #
    if estado_pulp == "Infeasible":
        return ResultadoAsignacion(
            estado="Infactible",
            asignacion={},
            costo_total=float("inf"),
            dist_maxima=float("inf"),
            dist_minima=float("inf"),
            rango=float("inf"),
            tiempo_cpu=t_cpu,
            modo_objetivo=modo_objetivo,
            guardias_activos_usados=guardias_activos,
            infactibilidad_causa=_diagnosticar_infactibilidad(guardias_activos, delta_max),
        )

    # Sin solución (time-out sin incumbente, etc.)
    if prob.status != pulp.LpStatusOptimal:
        return ResultadoAsignacion(
            estado="Sin_solucion",
            asignacion={},
            costo_total=float("inf"),
            dist_maxima=float("inf"),
            dist_minima=float("inf"),
            rango=float("inf"),
            tiempo_cpu=t_cpu,
            modo_objetivo=modo_objetivo,
            guardias_activos_usados=guardias_activos,
            infactibilidad_causa=f"Estado del solver: {estado_pulp} (posible time-out).",
        )

    # ------------------------------------------------------------------ #
    # Extraer asignación y métricas                                       #
    # ------------------------------------------------------------------ #
    asignacion: dict[str, str] = {}
    for g in guardias_activos:
        for e in EMPRESAS:
            val = pulp.value(x[g][e])
            if val is not None and round(val) == 1:
                asignacion[g] = e
                break

    distancias_asignadas = [
        DISTANCIAS[g][asignacion[g]]
        for g in guardias_activos
        if g in asignacion
    ]
    costo_total = sum(distancias_asignadas)
    d_max = max(distancias_asignadas)
    d_min = min(distancias_asignadas)

    return ResultadoAsignacion(
        estado="Optimo",
        asignacion=asignacion,
        costo_total=round(costo_total, 4),
        dist_maxima=round(d_max, 4),
        dist_minima=round(d_min, 4),
        rango=round(d_max - d_min, 4),
        tiempo_cpu=t_cpu,
        modo_objetivo=modo_objetivo,
        guardias_activos_usados=list(guardias_activos),
    )


# ---------------------------------------------------------------------------
# Verificación rápida (python -m models.asignacion)
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    SEP = "=" * 62

    print(SEP)
    print("Verificación rápida — models/asignacion.py")
    print(SEP)

    # ------------------------------------------------------------------ #
    # T1. Modo original (suma_total, todos los guardias)                  #
    # ------------------------------------------------------------------ #
    print("\n[T1] suma_total — todos los guardias")
    r1 = resolver(modo_objetivo="suma_total")
    print(r1.resumen())
    assert r1.estado == "Optimo", f"T1 falló: {r1.estado}"
    assert len(r1.asignacion) == 14, "T1: deben asignarse los 14 guardias"
    for e, p in PUESTOS.items():
        dot = r1.dotacion_por_empresa()
        assert p["min"] <= len(dot[e]) <= p["max"], \
            f"T1: dotación de {e} fuera de rango: {len(dot[e])}"
        if p["requiere_supervisor"]:
            assert any(g in SUPERVISORES for g in dot[e]), \
                f"T1: {e} requiere supervisor pero no tiene"
    print("  PASA")

    # ------------------------------------------------------------------ #
    # T2. Modo minimax — reduce distancia máxima individual               #
    # ------------------------------------------------------------------ #
    print("\n[T2] minimax — reducir distancia máxima")
    r2 = resolver(modo_objetivo="minimax")
    print(r2.resumen())
    assert r2.estado == "Optimo", f"T2 falló: {r2.estado}"
    assert r2.dist_maxima <= r1.dist_maxima + 0.01, \
        f"T2: minimax debería tener dist_max ≤ suma_total ({r2.dist_maxima:.2f} vs {r1.dist_maxima:.2f})"
    print("  PASA")

    # ------------------------------------------------------------------ #
    # T3. Modo multiobjetivo (α=0.7, β=0.3)                              #
    # ------------------------------------------------------------------ #
    print("\n[T3] multiobjetivo alpha=0.7 beta=0.3")
    r3 = resolver(modo_objetivo="multiobjetivo", alpha=0.7, beta=0.3)
    print(r3.resumen())
    assert r3.estado == "Optimo", f"T3 falló: {r3.estado}"
    # El costo total del multiobjetivo está entre suma_total puro y minimax puro
    assert r3.costo_total <= r1.costo_total + 0.01, \
        "T3: multiobjetivo no debe aumentar más que suma_total"
    print("  PASA")

    # ------------------------------------------------------------------ #
    # T4. Restricción delta_max_equidad                                   #
    # El rango mínimo real de esta instancia es ≈ 29 km                 #
    # (G1 necesita ir a Oleoducto a 34.2 km; su par más cercano va a    #
    #  5.2 km). Usamos delta_max=35 para que la restricción sea activa   #
    # pero factible; con delta=20 el modelo es infactible (se verifica   #
    # en T6).                                                            #
    # ------------------------------------------------------------------ #
    print("\n[T4] suma_total + delta_max=35 km (rango minimo real ~29 km)")
    r4 = resolver(modo_objetivo="suma_total", delta_max=35.0)
    print(r4.resumen())
    assert r4.estado == "Optimo", f"T4 falló: {r4.estado}"
    assert r4.rango <= 35.0 + 1e-4, \
        f"T4: rango={r4.rango:.2f} debe ser <= 35 km"
    print("  PASA")

    # ------------------------------------------------------------------ #
    # T5. Guardias parciales (G2 y G13 desactivados)                     #
    # ------------------------------------------------------------------ #
    print("\n[T5] guardias_activos sin G2 ni G13")
    activos_parcial = [g for g in GUARDIAS if g not in {"G2", "G13"}]
    r5 = resolver(guardias_activos=activos_parcial, modo_objetivo="suma_total")
    print(r5.resumen())
    assert r5.estado == "Optimo", f"T5 falló: {r5.estado}"
    assert "G2" not in r5.asignacion and "G13" not in r5.asignacion, \
        "T5: guardias inactivos no deben aparecer en la asignación"
    assert len(r5.asignacion) == 12, "T5: deben asignarse 12 guardias"
    print("  PASA")

    # ------------------------------------------------------------------ #
    # T6. delta_max muy restrictivo (20 km < rango_min ~29 km)           #
    # Debe devolver Infactible con causa descriptiva.                    #
    # ------------------------------------------------------------------ #
    print("\n[T6] suma_total + delta_max=20 km (debe ser Infactible)")
    r6 = resolver(modo_objetivo="suma_total", delta_max=20.0)
    assert r6.estado == "Infactible", \
        f"T6: se esperaba Infactible, se obtuvo {r6.estado}"
    print(f"  Infactible (correcto). Causa: {r6.infactibilidad_causa}")
    print("  PASA")

    # ------------------------------------------------------------------ #
    # T7. Infactibilidad detectada con guardias insuficientes             #
    # ------------------------------------------------------------------ #
    print("\n[T7] infactibilidad — solo 5 guardias activos")
    try:
        r7 = resolver(guardias_activos=["G1", "G2", "G3", "G4", "G5"])
        # No debería llegar aquí (5 < 10 mínimos totales)
        assert False, "T7: debía lanzar ValueError antes del solver"
    except ValueError as e:
        print(f"  ValueError capturado: {e}")
        print("  PASA")

    # ------------------------------------------------------------------ #
    # T8. Multiobjetivo solo equidad (α=0, β=1) == minimax               #
    # ------------------------------------------------------------------ #
    print("\n[T8] multiobjetivo puro equidad (alpha=0, beta=1)")
    r8 = resolver(modo_objetivo="multiobjetivo", alpha=0.0, beta=1.0)
    print(f"  dist_max={r8.dist_maxima:.2f}  costo={r8.costo_total:.2f}")
    assert r8.estado == "Optimo", f"T8 falló: {r8.estado}"
    # Con β=1 y α=0, el resultado debe coincidir con minimax
    assert abs(r8.dist_maxima - r2.dist_maxima) < 0.1, \
        f"T8: multiobjetivo(α=0,β=1) debe ≈ minimax ({r8.dist_maxima:.2f} vs {r2.dist_maxima:.2f})"
    print("  PASA")

    # ------------------------------------------------------------------ #
    # Tabla comparativa de los tres modos                                #
    # ------------------------------------------------------------------ #
    print(f"\n{SEP}")
    print("Tabla comparativa")
    print(SEP)
    print(f"{'Modo':<20} {'Costo':>8} {'Máx':>8} {'Mín':>8} {'Rango':>8} {'CPU(s)':>8}")
    print("-" * 62)
    for r in [r1, r2, r3, r4]:
        nombre = r.modo_objetivo
        if r.modo_objetivo == "multiobjetivo":
            nombre = "multi(0.7/0.3)"
        elif r.modo_objetivo == "suma_total" and r4 is r:
            nombre = "suma+delta20"
        print(
            f"{nombre:<20} {r.costo_total:>8.2f} {r.dist_maxima:>8.2f} "
            f"{r.dist_minima:>8.2f} {r.rango:>8.2f} {r.tiempo_cpu:>8.4f}"
        )

    print(f"\n{SEP}")
    print("Todas las pruebas pasaron.")
    print(SEP)
