"""tests/test_guard_manager.py — Tests pytest para GuardManager.

Cubre:
  · Alta de guardias (guardar_guardia).
  · Baja con distintos estados (desactivar_guardia).
  · Reactivación (reactivar_guardia).
  · Bloqueos de factibilidad: dotación mínima y supervisores.
  · Bloqueo de reactivación de guardias despedidos.
  · Validación de distancias (faltantes, no positivas).
  · Validación de IDs únicos y estados válidos.
  · Consultas: get_guardias_activos, get_todos_guardias, ids_activos.
  · Historial: acción, timestamp, motivo registrados.
  · Roundtrip CSV: exportar → importar preserva estado e historial.
  · Integración con solver (resolver_asignacion).
"""

from __future__ import annotations

import tempfile
from datetime import date
from pathlib import Path

import pytest

from app.components.guard_manager import (
    ESTADOS_VALIDOS,
    EstadoInvalido,
    Guardia,
    GuardManager,
    GuardiaNoEncontrado,
    OperacionBloqueada,
)
from models.asignacion import EMPRESAS, PUESTOS, SUPERVISORES

# ---------------------------------------------------------------------------
# Helpers y fixtures
# ---------------------------------------------------------------------------

EMPRESAS_TEST = ["Noramco", "ITI Chile", "Oleoducto", "Indama"]
PUESTOS_TEST = {
    "Noramco":   {"min": 3, "max": 5, "requiere_supervisor": True},
    "ITI Chile": {"min": 2, "max": 4, "requiere_supervisor": False},
    "Oleoducto": {"min": 3, "max": 5, "requiere_supervisor": True},
    "Indama":    {"min": 2, "max": 4, "requiere_supervisor": False},
}

#: Distancias base para guardia de prueba
_DIST_BASE = {"Noramco": 10.0, "ITI Chile": 12.0, "Oleoducto": 15.0, "Indama": 8.0}


def _make_gm(n_guards: int = 14, n_supervisors: int = 2) -> GuardManager:
    """Crea un GuardManager limpio con n guardias (primeros n_supervisors son supervisores)."""
    gm = GuardManager(empresas=EMPRESAS_TEST, puestos=PUESTOS_TEST)
    for i in range(1, n_guards + 1):
        gm.guardar_guardia(
            id=f"G{i}",
            nombre=f"Guardia {i}",
            es_supervisor=(i <= n_supervisors),
            distancias={e: float(i * 2 + j) for j, e in enumerate(EMPRESAS_TEST)},
        )
    return gm


@pytest.fixture()
def gm_completo() -> GuardManager:
    """GuardManager con 14 guardias activos (G1, G2 supervisores)."""
    return _make_gm(14, 2)


@pytest.fixture()
def gm_bursan() -> GuardManager:
    """GuardManager cargado desde la instancia Bursan real (models.asignacion)."""
    return GuardManager.desde_asignacion()


# ---------------------------------------------------------------------------
# Alta de guardias
# ---------------------------------------------------------------------------

class TestGuardarGuardia:

    def test_alta_basica(self, gm_completo):
        assert gm_completo.n_activos() == 14

    def test_guardia_aparece_en_activos(self, gm_completo):
        assert "G1" in gm_completo.ids_activos()

    def test_id_unico_raises(self, gm_completo):
        with pytest.raises(ValueError, match="ya existe"):
            gm_completo.guardar_guardia(
                "G1", "Otro", False, _DIST_BASE
            )

    def test_id_vacio_raises(self):
        gm = GuardManager(empresas=EMPRESAS_TEST, puestos=PUESTOS_TEST)
        with pytest.raises(ValueError, match="vacío"):
            gm.guardar_guardia("  ", "Nombre", False, _DIST_BASE)

    def test_estado_invalido_raises(self):
        gm = GuardManager(empresas=EMPRESAS_TEST, puestos=PUESTOS_TEST)
        with pytest.raises(EstadoInvalido):
            gm.guardar_guardia("G99", "Nombre", False, _DIST_BASE, estado="jubilado")

    def test_distancias_faltantes_raises(self):
        gm = GuardManager(empresas=EMPRESAS_TEST, puestos=PUESTOS_TEST)
        with pytest.raises(ValueError, match="faltan distancias"):
            gm.guardar_guardia("G99", "Nombre", False, {"Noramco": 10.0})

    def test_distancia_no_positiva_raises(self):
        gm = GuardManager(empresas=EMPRESAS_TEST, puestos=PUESTOS_TEST)
        dist_mala = {**_DIST_BASE, "Noramco": -5.0}
        with pytest.raises(ValueError, match="no positivas"):
            gm.guardar_guardia("G99", "Nombre", False, dist_mala)

    def test_distancia_cero_raises(self):
        gm = GuardManager(empresas=EMPRESAS_TEST, puestos=PUESTOS_TEST)
        dist_mala = {**_DIST_BASE, "Indama": 0.0}
        with pytest.raises(ValueError, match="no positivas"):
            gm.guardar_guardia("G99", "Nombre", False, dist_mala)

    def test_alta_registrada_en_historial(self, gm_completo):
        g = gm_completo.get_guardia("G1")
        assert len(g.historial) >= 1
        assert g.historial[0].accion == "alta"
        assert g.historial[0].estado_nuevo == "activo"

    def test_alta_con_estado_no_activo(self):
        gm = GuardManager(empresas=EMPRESAS_TEST, puestos=PUESTOS_TEST)
        gm.guardar_guardia("G99", "Nombre", False, _DIST_BASE, estado="licencia")
        assert gm.get_guardia("G99").estado == "licencia"
        assert not gm.get_guardia("G99").activo


# ---------------------------------------------------------------------------
# Desactivar guardia
# ---------------------------------------------------------------------------

class TestDesactivarGuardia:

    def test_desactivar_cambia_estado(self, gm_completo):
        gm_completo.desactivar_guardia("G5", "Vacaciones")
        assert gm_completo.get_guardia("G5").estado == "fuera_de_servicio"

    def test_desactivado_no_aparece_en_activos(self, gm_completo):
        gm_completo.desactivar_guardia("G5", "Test")
        assert "G5" not in gm_completo.ids_activos()

    def test_desactivar_con_estado_especifico(self, gm_completo):
        gm_completo.desactivar_guardia("G5", "Vacaciones anuales", nuevo_estado="vacaciones")
        assert gm_completo.get_guardia("G5").estado == "vacaciones"

    def test_desactivar_con_fecha_retorno_str(self, gm_completo):
        gm_completo.desactivar_guardia(
            "G5", "Licencia medica",
            nuevo_estado="licencia", fecha_retorno="2025-09-15"
        )
        assert gm_completo.get_guardia("G5").fecha_retorno_esperada == date(2025, 9, 15)

    def test_desactivar_con_fecha_retorno_date(self, gm_completo):
        gm_completo.desactivar_guardia(
            "G5", "Vacaciones",
            nuevo_estado="vacaciones",
            fecha_retorno=date(2025, 8, 31)
        )
        assert gm_completo.get_guardia("G5").fecha_retorno_esperada == date(2025, 8, 31)

    def test_desactivar_estado_activo_como_destino_raises(self, gm_completo):
        with pytest.raises(EstadoInvalido, match="reactivar_guardia"):
            gm_completo.desactivar_guardia("G5", "Test", nuevo_estado="activo")

    def test_desactivar_id_inexistente_raises(self, gm_completo):
        with pytest.raises(GuardiaNoEncontrado):
            gm_completo.desactivar_guardia("G99", "Test")

    def test_desactivar_ya_inactivo_raises(self, gm_completo):
        gm_completo.desactivar_guardia("G5", "Primera baja")
        with pytest.raises(ValueError, match="ya está en estado"):
            gm_completo.desactivar_guardia("G5", "Segunda baja")

    def test_desactivar_estado_invalido_raises(self, gm_completo):
        with pytest.raises(EstadoInvalido):
            gm_completo.desactivar_guardia("G5", "Test", nuevo_estado="no_existe")

    def test_baja_registrada_en_historial(self, gm_completo):
        gm_completo.desactivar_guardia("G5", "Vacaciones anuales", nuevo_estado="vacaciones")
        g = gm_completo.get_guardia("G5")
        bajas = [h for h in g.historial if h.accion == "baja"]
        assert len(bajas) == 1
        assert bajas[0].estado_anterior == "activo"
        assert bajas[0].estado_nuevo == "vacaciones"
        assert "Vacaciones" in bajas[0].motivo
        assert bajas[0].timestamp is not None

    @pytest.mark.parametrize("estado", ["vacaciones", "fuera_de_servicio", "despedido", "licencia"])
    def test_todos_estados_inactivos_aceptados(self, estado):
        gm = _make_gm(14, 2)
        gm.desactivar_guardia("G5", "Test", nuevo_estado=estado)
        assert gm.get_guardia("G5").estado == estado


# ---------------------------------------------------------------------------
# Bloqueos de factibilidad
# ---------------------------------------------------------------------------

class TestBloqueoFactibilidad:

    def test_bloqueo_dotacion_minima(self, gm_completo):
        """No se puede bajar un guardia si ya se está en el mínimo."""
        total_min = gm_completo.total_min_dotacion  # 10
        # Desactivar guardias no-supervisores hasta llegar exactamente al mínimo
        no_sups = [g for g in gm_completo.ids_activos()
                   if not gm_completo.get_guardia(g).es_supervisor]
        excedente = gm_completo.n_activos() - total_min
        for g_id in no_sups[:excedente]:
            gm_completo.desactivar_guardia(g_id, "Reduccion test")
        assert gm_completo.n_activos() == total_min

        candidato = next(g for g in gm_completo.ids_activos()
                         if not gm_completo.get_guardia(g).es_supervisor)
        with pytest.raises(OperacionBloqueada, match="guardias activos"):
            gm_completo.desactivar_guardia(candidato, "Baja que debe bloquearse")

    def test_bloqueo_supervisor_para_empresa_que_lo_requiere(self, gm_bursan):
        """No se puede bajar G1 (supervisor) si G6 es el único otro supervisor
        y hay 2 empresas que requieren supervisor."""
        # Noramco y Oleoducto requieren supervisor; hay 2 supervisores (G1, G6).
        # Bajar cualquiera de los dos deja 1 supervisor para 2 empresas → bloqueo.
        with pytest.raises(OperacionBloqueada, match="supervisor"):
            gm_bursan.desactivar_guardia("G1", "Test bloqueo supervisor")

    def test_bloqueo_supervisor_g6(self, gm_bursan):
        with pytest.raises(OperacionBloqueada, match="supervisor"):
            gm_bursan.desactivar_guardia("G6", "Test bloqueo supervisor")

    def test_sin_bloqueo_con_supervisor_extra(self):
        """Con 3 supervisores para 2 empresas, se puede bajar uno."""
        gm = GuardManager(empresas=EMPRESAS_TEST, puestos=PUESTOS_TEST)
        for i in range(1, 14):
            gm.guardar_guardia(
                f"G{i}", f"G{i}", es_supervisor=(i <= 3),
                distancias=_DIST_BASE,
            )
        # 3 supervisores, 2 empresas con supervisor req → puede bajar 1
        gm.desactivar_guardia("G3", "Test con supervisor extra")
        assert gm.get_guardia("G3").estado != "activo"

    def test_mensaje_error_dotacion_es_descriptivo(self, gm_completo):
        no_sups = [g for g in gm_completo.ids_activos()
                   if not gm_completo.get_guardia(g).es_supervisor]
        excedente = gm_completo.n_activos() - gm_completo.total_min_dotacion
        for g_id in no_sups[:excedente]:
            gm_completo.desactivar_guardia(g_id, "Reduccion")
        candidato = next(g for g in gm_completo.ids_activos()
                         if not gm_completo.get_guardia(g).es_supervisor)
        with pytest.raises(OperacionBloqueada) as exc_info:
            gm_completo.desactivar_guardia(candidato, "Baja bloqueada")
        msg = str(exc_info.value)
        assert "guardias activos" in msg.lower() or "activa" in msg.lower()


# ---------------------------------------------------------------------------
# Reactivar guardia
# ---------------------------------------------------------------------------

class TestReactivarGuardia:

    def test_reactivar_cambia_estado_a_activo(self, gm_completo):
        gm_completo.desactivar_guardia("G5", "Baja temporal")
        gm_completo.reactivar_guardia("G5", "Regreso")
        assert gm_completo.get_guardia("G5").activo

    def test_reactivado_aparece_en_activos(self, gm_completo):
        gm_completo.desactivar_guardia("G5", "Baja temporal")
        gm_completo.reactivar_guardia("G5")
        assert "G5" in gm_completo.ids_activos()

    def test_reactivar_limpia_fecha_retorno(self, gm_completo):
        gm_completo.desactivar_guardia("G5", "Vacaciones",
                                        nuevo_estado="vacaciones",
                                        fecha_retorno="2025-08-01")
        gm_completo.reactivar_guardia("G5")
        assert gm_completo.get_guardia("G5").fecha_retorno_esperada is None

    def test_reactivar_despedido_bloqueado(self, gm_completo):
        gm_completo.desactivar_guardia("G5", "Conducta inapropiada",
                                        nuevo_estado="despedido")
        with pytest.raises(OperacionBloqueada, match="despedido"):
            gm_completo.reactivar_guardia("G5")

    def test_reactivar_ya_activo_raises(self, gm_completo):
        with pytest.raises(ValueError, match="ya est"):
            gm_completo.reactivar_guardia("G5")

    def test_reactivar_id_inexistente_raises(self, gm_completo):
        with pytest.raises(GuardiaNoEncontrado):
            gm_completo.reactivar_guardia("G99")

    def test_reactivacion_registrada_en_historial(self, gm_completo):
        gm_completo.desactivar_guardia("G5", "Baja temporal", nuevo_estado="licencia")
        gm_completo.reactivar_guardia("G5", "Alta medica")
        g = gm_completo.get_guardia("G5")
        react = [h for h in g.historial if h.accion == "reactivacion"]
        assert len(react) == 1
        assert react[0].estado_anterior == "licencia"
        assert react[0].estado_nuevo == "activo"
        assert "Alta medica" in react[0].motivo

    @pytest.mark.parametrize("estado_previo", ["vacaciones", "fuera_de_servicio", "licencia"])
    def test_reactivar_desde_cualquier_estado_temporal(self, estado_previo):
        gm = _make_gm(14, 2)
        gm.desactivar_guardia("G5", "Test", nuevo_estado=estado_previo)
        gm.reactivar_guardia("G5")
        assert gm.get_guardia("G5").activo


# ---------------------------------------------------------------------------
# Consultas
# ---------------------------------------------------------------------------

class TestConsultas:

    def test_get_guardias_activos_solo_activos(self, gm_completo):
        gm_completo.desactivar_guardia("G5", "Test")
        activos = gm_completo.get_guardias_activos()
        assert all(g.activo for g in activos)
        assert not any(g.id == "G5" for g in activos)

    def test_get_todos_incluye_inactivos(self, gm_completo):
        gm_completo.desactivar_guardia("G5", "Test")
        todos = gm_completo.get_todos_guardias()
        assert len(todos) == 14
        g5 = next(g for g in todos if g.id == "G5")
        assert not g5.activo

    def test_ids_activos_ordenados(self, gm_completo):
        ids = gm_completo.ids_activos()
        assert ids == sorted(ids)

    def test_get_guardia_inexistente_raises(self, gm_completo):
        with pytest.raises(GuardiaNoEncontrado):
            gm_completo.get_guardia("G99")

    def test_n_activos_decrementa_tras_baja(self, gm_completo):
        n_before = gm_completo.n_activos()
        gm_completo.desactivar_guardia("G5", "Test")
        assert gm_completo.n_activos() == n_before - 1

    def test_n_activos_incrementa_tras_reactivacion(self, gm_completo):
        gm_completo.desactivar_guardia("G5", "Test")
        n_before = gm_completo.n_activos()
        gm_completo.reactivar_guardia("G5")
        assert gm_completo.n_activos() == n_before + 1

    def test_n_supervisores_activos(self, gm_completo):
        assert gm_completo.n_supervisores_activos() == 2

    def test_distancias_activos_estructura(self, gm_completo):
        d = gm_completo.distancias_activos()
        assert set(d.keys()) == set(gm_completo.ids_activos())
        for gid, dist in d.items():
            assert set(dist.keys()) == set(EMPRESAS_TEST)

    def test_es_factible_verdadero(self, gm_completo):
        assert gm_completo.es_factible()

    def test_resumen_contiene_activos(self, gm_completo):
        txt = gm_completo.resumen()
        assert "14" in txt or "Activos" in txt


# ---------------------------------------------------------------------------
# Historial
# ---------------------------------------------------------------------------

class TestHistorial:

    def test_historial_ordenado_cronologicamente(self, gm_completo):
        gm_completo.desactivar_guardia("G5", "Baja")
        gm_completo.reactivar_guardia("G5", "Regreso")
        h = gm_completo.get_guardia("G5").historial
        timestamps = [e.timestamp for e in h]
        assert timestamps == sorted(timestamps)

    def test_historial_tres_entradas_ciclo_completo(self, gm_completo):
        gm_completo.desactivar_guardia("G5", "Baja")
        gm_completo.reactivar_guardia("G5")
        h = gm_completo.get_guardia("G5").historial
        acciones = [e.accion for e in h]
        assert "alta" in acciones
        assert "baja" in acciones
        assert "reactivacion" in acciones

    def test_historial_dataframe_columnas(self, gm_completo):
        gm_completo.desactivar_guardia("G5", "Baja test")
        df = gm_completo.historial_dataframe()
        required = {"guardia_id", "timestamp", "accion", "estado_anterior",
                    "estado_nuevo", "motivo", "fecha_retorno"}
        assert required.issubset(df.columns)

    def test_historial_dataframe_no_vacio(self, gm_completo):
        df = gm_completo.historial_dataframe()
        assert len(df) >= 14  # al menos el alta de cada guardia

    def test_fecha_retorno_en_historial(self, gm_completo):
        gm_completo.desactivar_guardia("G5", "Vacaciones",
                                        nuevo_estado="vacaciones",
                                        fecha_retorno="2025-08-15")
        bajas = [h for h in gm_completo.get_guardia("G5").historial
                 if h.accion == "baja"]
        assert bajas[0].fecha_retorno == date(2025, 8, 15)


# ---------------------------------------------------------------------------
# CSV roundtrip
# ---------------------------------------------------------------------------

class TestCSVRoundtrip:

    def test_exportar_crea_archivo(self, gm_completo, tmp_path):
        csv_path = tmp_path / "guardias.csv"
        gm_completo.exportar_csv(csv_path)
        assert csv_path.exists()

    def test_exportar_crea_historial(self, gm_completo, tmp_path):
        csv_path = tmp_path / "guardias.csv"
        gm_completo.exportar_csv(csv_path)
        assert (tmp_path / "guardias_historial.csv").exists()

    def test_importar_restaura_cantidad_guardias(self, gm_completo, tmp_path):
        csv_path = tmp_path / "guardias.csv"
        gm_completo.exportar_csv(csv_path)

        gm2 = GuardManager(empresas=EMPRESAS_TEST, puestos=PUESTOS_TEST)
        gm2.importar_csv(csv_path)
        assert gm2.n_activos() == gm_completo.n_activos()

    def test_importar_restaura_estados(self, gm_completo, tmp_path):
        gm_completo.desactivar_guardia("G5", "Vacaciones", nuevo_estado="vacaciones")
        csv_path = tmp_path / "guardias.csv"
        gm_completo.exportar_csv(csv_path)

        gm2 = GuardManager(empresas=EMPRESAS_TEST, puestos=PUESTOS_TEST)
        gm2.importar_csv(csv_path)
        assert gm2.get_guardia("G5").estado == "vacaciones"

    def test_importar_restaura_historial(self, gm_completo, tmp_path):
        gm_completo.desactivar_guardia("G5", "Baja test")
        gm_completo.reactivar_guardia("G5")
        csv_path = tmp_path / "guardias.csv"
        gm_completo.exportar_csv(csv_path)

        gm2 = GuardManager(empresas=EMPRESAS_TEST, puestos=PUESTOS_TEST)
        gm2.importar_csv(csv_path)
        g5_orig = gm_completo.get_guardia("G5")
        g5_rest = gm2.get_guardia("G5")
        assert len(g5_rest.historial) == len(g5_orig.historial)

    def test_importar_restaura_fecha_retorno(self, gm_completo, tmp_path):
        gm_completo.desactivar_guardia("G5", "Vacaciones",
                                        nuevo_estado="vacaciones",
                                        fecha_retorno="2025-08-31")
        csv_path = tmp_path / "guardias.csv"
        gm_completo.exportar_csv(csv_path)

        gm2 = GuardManager(empresas=EMPRESAS_TEST, puestos=PUESTOS_TEST)
        gm2.importar_csv(csv_path)
        assert gm2.get_guardia("G5").fecha_retorno_esperada == date(2025, 8, 31)

    def test_importar_restaura_distancias(self, gm_completo, tmp_path):
        csv_path = tmp_path / "guardias.csv"
        gm_completo.exportar_csv(csv_path)

        gm2 = GuardManager(empresas=EMPRESAS_TEST, puestos=PUESTOS_TEST)
        gm2.importar_csv(csv_path)
        for g in gm_completo.get_todos_guardias():
            g_rest = gm2.get_guardia(g.id)
            for emp in EMPRESAS_TEST:
                assert g_rest.distancias.get(emp) == pytest.approx(
                    g.distancias[emp], rel=1e-4
                )

    def test_importar_archivo_inexistente_raises(self, tmp_path):
        gm = GuardManager(empresas=EMPRESAS_TEST, puestos=PUESTOS_TEST)
        with pytest.raises(FileNotFoundError):
            gm.importar_csv(tmp_path / "no_existe.csv")

    def test_ids_activos_iguales_tras_roundtrip(self, gm_completo, tmp_path):
        gm_completo.desactivar_guardia("G5", "Test")
        csv_path = tmp_path / "guardias.csv"
        gm_completo.exportar_csv(csv_path)

        gm2 = GuardManager(empresas=EMPRESAS_TEST, puestos=PUESTOS_TEST)
        gm2.importar_csv(csv_path)
        assert gm2.ids_activos() == gm_completo.ids_activos()


# ---------------------------------------------------------------------------
# Integración con el solver
# ---------------------------------------------------------------------------

class TestIntegracionSolver:

    def test_resolver_asignacion_optimo(self, gm_bursan):
        r = gm_bursan.resolver_asignacion(modo_objetivo="suma_total")
        assert r.estado == "Optimo"

    def test_resolver_minimax(self, gm_bursan):
        r = gm_bursan.resolver_asignacion(modo_objetivo="minimax")
        assert r.estado == "Optimo"

    def test_resolver_sin_guardia_inactivo(self, gm_bursan):
        gm_bursan.desactivar_guardia("G14", "Test", nuevo_estado="licencia")
        r = gm_bursan.resolver_asignacion()
        assert r.estado == "Optimo"
        assert "G14" not in r.asignacion

    def test_desde_asignacion_carga_14_guardias(self):
        gm = GuardManager.desde_asignacion()
        assert gm.n_activos() == 14

    def test_desde_asignacion_detecta_supervisores(self):
        gm = GuardManager.desde_asignacion()
        sups = [g for g in gm.get_guardias_activos() if g.es_supervisor]
        assert len(sups) == 2
        sup_ids = {g.id for g in sups}
        assert sup_ids == SUPERVISORES
