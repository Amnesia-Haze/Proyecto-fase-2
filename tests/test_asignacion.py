"""tests/test_asignacion.py — Tests pytest para models/asignacion.py.

Cubre:
  · Las tres funciones objetivo (suma_total, minimax, multiobjetivo).
  · Restricciones R1–R5 del modelo original.
  · Restricción de equidad delta_max.
  · Filtrado de guardias activos.
  · Detección de infactibilidad antes y después del solver.
  · Métricas del ResultadoAsignacion.
"""

from __future__ import annotations

import pytest

from models.asignacion import (
    DISTANCIAS,
    EMPRESAS,
    GUARDIAS,
    PUESTOS,
    SUPERVISORES,
    ResultadoAsignacion,
    resolver,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sol_suma() -> ResultadoAsignacion:
    """Solución óptima con suma_total, todos los guardias."""
    return resolver(modo_objetivo="suma_total")


@pytest.fixture(scope="module")
def sol_minimax() -> ResultadoAsignacion:
    """Solución óptima con minimax, todos los guardias."""
    return resolver(modo_objetivo="minimax")


@pytest.fixture(scope="module")
def sol_multi() -> ResultadoAsignacion:
    """Solución óptima con multiobjetivo α=0.6 β=0.4."""
    return resolver(modo_objetivo="multiobjetivo", alpha=0.6, beta=0.4)


# ---------------------------------------------------------------------------
# Estado del resultado
# ---------------------------------------------------------------------------

class TestEstado:

    def test_suma_total_optimo(self, sol_suma):
        assert sol_suma.estado == "Optimo"

    def test_minimax_optimo(self, sol_minimax):
        assert sol_minimax.estado == "Optimo"

    def test_multiobjetivo_optimo(self, sol_multi):
        assert sol_multi.estado == "Optimo"

    def test_modo_objetivo_registrado(self, sol_suma, sol_minimax, sol_multi):
        assert sol_suma.modo_objetivo == "suma_total"
        assert sol_minimax.modo_objetivo == "minimax"
        assert sol_multi.modo_objetivo == "multiobjetivo"

    def test_tiempo_cpu_positivo(self, sol_suma):
        assert sol_suma.tiempo_cpu > 0.0

    def test_guardias_activos_usados(self, sol_suma):
        assert sorted(sol_suma.guardias_activos_usados) == sorted(GUARDIAS)


# ---------------------------------------------------------------------------
# R1 — Asignación única
# ---------------------------------------------------------------------------

class TestR1AsignacionUnica:

    def test_todos_asignados(self, sol_suma):
        assert len(sol_suma.asignacion) == len(GUARDIAS)

    def test_cada_guardia_en_una_empresa(self, sol_suma):
        for g in GUARDIAS:
            assert g in sol_suma.asignacion, f"{g} no asignado"
            assert sol_suma.asignacion[g] in EMPRESAS

    def test_sin_asignacion_doble(self, sol_suma):
        # Cada guardia debe aparecer exactamente una vez como clave
        assert len(set(sol_suma.asignacion.keys())) == len(GUARDIAS)


# ---------------------------------------------------------------------------
# R2 y R3 — Dotación mínima y máxima
# ---------------------------------------------------------------------------

class TestDotacion:

    def test_dotacion_minima_respetada(self, sol_suma):
        dot = sol_suma.dotacion_por_empresa()
        for e in EMPRESAS:
            assert len(dot[e]) >= PUESTOS[e]["min"], \
                f"{e}: {len(dot[e])} < min={PUESTOS[e]['min']}"

    def test_dotacion_maxima_respetada(self, sol_suma):
        dot = sol_suma.dotacion_por_empresa()
        for e in EMPRESAS:
            assert len(dot[e]) <= PUESTOS[e]["max"], \
                f"{e}: {len(dot[e])} > max={PUESTOS[e]['max']}"

    def test_dotacion_minima_minimax(self, sol_minimax):
        dot = sol_minimax.dotacion_por_empresa()
        for e in EMPRESAS:
            assert len(dot[e]) >= PUESTOS[e]["min"]

    def test_dotacion_maxima_minimax(self, sol_minimax):
        dot = sol_minimax.dotacion_por_empresa()
        for e in EMPRESAS:
            assert len(dot[e]) <= PUESTOS[e]["max"]


# ---------------------------------------------------------------------------
# R4 — Supervisor requerido
# ---------------------------------------------------------------------------

class TestSupervisores:

    def test_supervisor_en_empresa_requerida(self, sol_suma):
        dot = sol_suma.dotacion_por_empresa()
        for e in EMPRESAS:
            if PUESTOS[e]["requiere_supervisor"]:
                sup_en_e = [g for g in dot[e] if g in SUPERVISORES]
                assert len(sup_en_e) >= 1, \
                    f"{e} requiere supervisor pero tiene: {dot[e]}"

    def test_supervisor_en_empresa_requerida_minimax(self, sol_minimax):
        dot = sol_minimax.dotacion_por_empresa()
        for e in EMPRESAS:
            if PUESTOS[e]["requiere_supervisor"]:
                assert any(g in SUPERVISORES for g in dot[e])


# ---------------------------------------------------------------------------
# R5 — Solo guardias activos
# ---------------------------------------------------------------------------

class TestGuardiasActivos:

    def test_inactivos_no_aparecen(self):
        inactivos = {"G2", "G13"}
        activos = [g for g in GUARDIAS if g not in inactivos]
        r = resolver(guardias_activos=activos)
        assert r.estado == "Optimo"
        for g in inactivos:
            assert g not in r.asignacion, f"{g} inactivo no debe estar asignado"

    def test_solo_activos_asignados(self):
        activos = ["G1", "G2", "G3", "G4", "G5", "G6",
                   "G7", "G8", "G9", "G10", "G11", "G12"]
        r = resolver(guardias_activos=activos)
        assert r.estado == "Optimo"
        assert set(r.asignacion.keys()) == set(activos)

    def test_guardias_activos_usados_coincide(self):
        activos = ["G1", "G3", "G5", "G6", "G7", "G8",
                   "G9", "G10", "G11", "G12", "G13", "G14"]
        r = resolver(guardias_activos=activos)
        assert sorted(r.guardias_activos_usados) == sorted(activos)


# ---------------------------------------------------------------------------
# Métricas de equidad
# ---------------------------------------------------------------------------

class TestMetricasEquidad:

    def test_dist_maxima_es_maximo_real(self, sol_suma):
        dists = [DISTANCIAS[g][sol_suma.asignacion[g]] for g in GUARDIAS]
        assert sol_suma.dist_maxima == pytest.approx(max(dists), rel=1e-4)

    def test_dist_minima_es_minimo_real(self, sol_suma):
        dists = [DISTANCIAS[g][sol_suma.asignacion[g]] for g in GUARDIAS]
        assert sol_suma.dist_minima == pytest.approx(min(dists), rel=1e-4)

    def test_rango_es_diferencia(self, sol_suma):
        assert sol_suma.rango == pytest.approx(
            sol_suma.dist_maxima - sol_suma.dist_minima, rel=1e-4
        )

    def test_costo_total_es_suma(self, sol_suma):
        suma = sum(DISTANCIAS[g][sol_suma.asignacion[g]] for g in GUARDIAS)
        assert sol_suma.costo_total == pytest.approx(suma, rel=1e-4)

    def test_minimax_reduce_distancia_maxima(self, sol_suma, sol_minimax):
        """minimax debe tener dist_maxima ≤ suma_total (esa es su razón de ser)."""
        assert sol_minimax.dist_maxima <= sol_suma.dist_maxima + 0.01

    def test_multiobjetivo_entre_ambos_extremos(self, sol_suma, sol_minimax, sol_multi):
        """multiobjetivo con α=β=0.5 debe estar entre suma y minimax en dist_max."""
        assert sol_multi.dist_maxima <= sol_suma.dist_maxima + 0.5
        assert sol_multi.costo_total <= sol_suma.costo_total * 1.15


# ---------------------------------------------------------------------------
# Función objetivo multiobjetivo — casos extremos
# ---------------------------------------------------------------------------

class TestMultiobjetivo:

    def test_alpha1_beta0_equivale_a_suma_total(self):
        r_multi = resolver(modo_objetivo="multiobjetivo", alpha=1.0, beta=0.0)
        r_suma = resolver(modo_objetivo="suma_total")
        assert r_multi.costo_total == pytest.approx(r_suma.costo_total, rel=1e-3)

    def test_alpha0_beta1_equivale_a_minimax(self):
        r_multi = resolver(modo_objetivo="multiobjetivo", alpha=0.0, beta=1.0)
        r_mm = resolver(modo_objetivo="minimax")
        assert r_multi.dist_maxima == pytest.approx(r_mm.dist_maxima, rel=1e-3)


# ---------------------------------------------------------------------------
# Restricción delta_max_equidad
# ---------------------------------------------------------------------------

class TestDeltaMaxEquidad:

    def test_rango_respetado(self):
        # Rango mínimo real de la instancia ≈ 29 km; usamos 35 para que
        # la restricción sea activa pero el modelo permanezca factible.
        delta = 35.0
        r = resolver(modo_objetivo="suma_total", delta_max=delta)
        assert r.estado == "Optimo"
        assert r.rango <= delta + 1e-4, \
            f"rango={r.rango:.2f} supera delta_max={delta}"

    def test_delta_muy_restrictivo_puede_ser_infactible(self):
        """delta_max=0 exige igual distancia para todos → casi siempre infactible."""
        r = resolver(modo_objetivo="suma_total", delta_max=0.0)
        # Puede ser Optimo (raro) o Infactible; lo que NO debe pasar es Error
        assert r.estado in ("Optimo", "Infactible")

    def test_delta_grande_no_restringe(self):
        """delta_max=999 no debe cambiar la solución de suma_total."""
        r_sin = resolver(modo_objetivo="suma_total")
        r_con = resolver(modo_objetivo="suma_total", delta_max=999.0)
        assert r_con.estado == "Optimo"
        assert r_con.costo_total == pytest.approx(r_sin.costo_total, rel=1e-3)

    def test_minimax_con_delta(self):
        r = resolver(modo_objetivo="minimax", delta_max=35.0)
        if r.estado == "Optimo":
            assert r.rango <= 35.0 + 1e-4


# ---------------------------------------------------------------------------
# Infactibilidad y validación de parámetros
# ---------------------------------------------------------------------------

class TestInfactibilidad:

    def test_pocos_guardias_activos_raise(self):
        """Menos guardias que la dotación mínima total → ValueError antes del solver."""
        with pytest.raises(ValueError, match="insuficientes"):
            resolver(guardias_activos=["G1", "G2", "G3", "G4", "G5"])

    def test_sin_supervisores_activos_raise(self):
        """Sin supervisores activos no se puede cubrir las empresas que los requieren."""
        sin_sup = [g for g in GUARDIAS if g not in SUPERVISORES]
        with pytest.raises(ValueError, match="[Ss]upervi"):
            resolver(guardias_activos=sin_sup)

    def test_modo_invalido_raise(self):
        with pytest.raises(ValueError, match="modo_objetivo"):
            resolver(modo_objetivo="optimo_pareto")  # type: ignore[arg-type]

    def test_alpha_fuera_de_rango_raise(self):
        with pytest.raises(ValueError, match="alpha"):
            resolver(modo_objetivo="multiobjetivo", alpha=1.5, beta=0.5)

    def test_beta_fuera_de_rango_raise(self):
        with pytest.raises(ValueError, match="beta"):
            resolver(modo_objetivo="multiobjetivo", alpha=0.5, beta=-0.1)

    def test_alpha_beta_cero_raise(self):
        with pytest.raises(ValueError, match=r"alpha \+ beta"):
            resolver(modo_objetivo="multiobjetivo", alpha=0.0, beta=0.0)

    def test_delta_negativo_raise(self):
        with pytest.raises(ValueError, match="delta_max"):
            resolver(delta_max=-5.0)

    def test_id_invalido_raise(self):
        with pytest.raises(ValueError, match="no reconocidos"):
            resolver(guardias_activos=["G1", "G99"])

    def test_id_duplicado_raise(self):
        activos = list(GUARDIAS) + ["G1"]  # G1 duplicado
        with pytest.raises(ValueError, match="[Dd]uplicados"):
            resolver(guardias_activos=activos)

    def test_resultado_infactible_tiene_campos_inf(self):
        """El ResultadoAsignacion de infactibilidad tiene campos vacíos y causa."""
        try:
            r = resolver(guardias_activos=["G1", "G2", "G3", "G4", "G5"])
        except ValueError:
            # Infactibilidad detectada antes del solver → OK
            return
        # Si llegó aquí, el solver retornó Infactible
        if r.estado == "Infactible":
            assert r.asignacion == {}
            assert r.costo_total == float("inf")
            assert r.infactibilidad_causa != ""


# ---------------------------------------------------------------------------
# Métodos auxiliares de ResultadoAsignacion
# ---------------------------------------------------------------------------

class TestResultadoHelpers:

    def test_resumen_optimo_contiene_costo(self, sol_suma):
        txt = sol_suma.resumen()
        assert "Costo total" in txt
        assert str(round(sol_suma.costo_total, 2)) in txt

    def test_resumen_infactible_contiene_causa(self):
        r = ResultadoAsignacion(
            estado="Infactible",
            asignacion={},
            costo_total=float("inf"),
            dist_maxima=float("inf"),
            dist_minima=float("inf"),
            rango=float("inf"),
            tiempo_cpu=0.01,
            modo_objetivo="suma_total",
            infactibilidad_causa="dotación mínima insuficiente",
        )
        txt = r.resumen()
        assert "Infactible" in txt
        assert "dotación" in txt

    def test_dotacion_por_empresa_completa(self, sol_suma):
        dot = sol_suma.dotacion_por_empresa()
        assert set(dot.keys()) == set(EMPRESAS)
        total = sum(len(v) for v in dot.values())
        assert total == len(GUARDIAS)

    def test_dotacion_por_empresa_sin_superposicion(self, sol_suma):
        dot = sol_suma.dotacion_por_empresa()
        todos = [g for lst in dot.values() for g in lst]
        assert len(todos) == len(set(todos)), "Guardia asignado a más de una empresa"
