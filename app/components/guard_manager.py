"""app/components/guard_manager.py — Gestión dinámica de guardias Bursan.

Responsabilidades
-----------------
* Ciclo de vida completo: alta, baja temporal/permanente, reactivación.
* Validación de integridad de datos (distancias, IDs únicos, estados).
* Guardia de factibilidad: bloquea bajas que harían infactible el modelo ILP
  (dotación mínima y supervisores por empresa).
* Historial inmutable de cambios con timestamp para auditoría.
* Exportación e importación CSV con restauración completa del historial.

Integración con el solver
-------------------------
    gm = GuardManager.desde_asignacion()   # carga la instancia Bursan completa
    gm.desactivar_guardia("G3", "Vacaciones", nuevo_estado="vacaciones",
                           fecha_retorno="2025-08-15")
    resultado = gm.resolver_asignacion(modo_objetivo="minimax")

Ejecución rápida
----------------
    python -m app.components.guard_manager
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Literal, Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Tipos
# ---------------------------------------------------------------------------

EstadoGuardia = Literal[
    "activo", "vacaciones", "fuera_de_servicio", "despedido", "licencia"
]

ESTADOS_VALIDOS: frozenset[str] = frozenset({
    "activo", "vacaciones", "fuera_de_servicio", "despedido", "licencia"
})

#: Estados desde los que se puede reactivar.
ESTADOS_REACTIVABLES: frozenset[str] = frozenset({
    "vacaciones", "fuera_de_servicio", "licencia"
})

# ---------------------------------------------------------------------------
# Exceciones específicas del dominio
# ---------------------------------------------------------------------------


class OperacionBloqueada(Exception):
    """Una operación violaría una restricción de factibilidad del modelo ILP."""


class GuardiaNoEncontrado(KeyError):
    """Se referencia un ID de guardia que no existe en el registro."""


class EstadoInvalido(ValueError):
    """Se intenta asignar un estado no reconocido por el sistema."""


# ---------------------------------------------------------------------------
# Modelos de datos
# ---------------------------------------------------------------------------


@dataclass
class EntradaHistorial:
    """Registro inmutable de un cambio de estado de un guardia.

    Attributes
    ----------
    timestamp : datetime
        Momento exacto del cambio.
    accion : str
        ``'alta'``, ``'baja'`` o ``'reactivacion'``.
    estado_anterior : str
        Estado previo al cambio (vacío en el alta).
    estado_nuevo : str
        Estado resultante.
    motivo : str
        Texto libre que explica la razón del cambio.
    fecha_retorno : date | None
        Fecha estimada de regreso (solo para bajas temporales).
    """

    timestamp: datetime
    accion: str
    estado_anterior: str
    estado_nuevo: str
    motivo: str
    fecha_retorno: Optional[date]

    def to_dict(self, guardia_id: str) -> dict:
        """Serializa a dict plano para exportar a CSV."""
        return {
            "guardia_id": guardia_id,
            "timestamp": self.timestamp.isoformat(),
            "accion": self.accion,
            "estado_anterior": self.estado_anterior,
            "estado_nuevo": self.estado_nuevo,
            "motivo": self.motivo,
            "fecha_retorno": self.fecha_retorno.isoformat() if self.fecha_retorno else "",
        }


@dataclass
class Guardia:
    """Guardia de seguridad con su ciclo de vida completo.

    Attributes
    ----------
    id : str
        Identificador único (ej. ``'G1'``).
    nombre : str
        Nombre completo.
    es_supervisor : bool
        True si tiene certificación de supervisor.
    distancias : dict[str, float]
        Distancias en km a cada empresa destino.
    estado : str
        Estado actual; uno de ``ESTADOS_VALIDOS``.
    fecha_ingreso : date
        Fecha de incorporación al sistema.
    fecha_retorno_esperada : date | None
        Fecha de regreso prevista para bajas temporales.
    historial : list[EntradaHistorial]
        Historial completo de cambios de estado, en orden cronológico.
    """

    id: str
    nombre: str
    es_supervisor: bool
    distancias: dict[str, float]
    estado: str = "activo"
    fecha_ingreso: date = field(default_factory=date.today)
    fecha_retorno_esperada: Optional[date] = None
    historial: list[EntradaHistorial] = field(default_factory=list)

    @property
    def activo(self) -> bool:
        """True si el guardia puede participar en el modelo ILP."""
        return self.estado == "activo"

    def to_dict(self, empresas: list[str]) -> dict:
        """Serializa a dict plano para exportar a CSV."""
        row: dict = {
            "id": self.id,
            "nombre": self.nombre,
            "es_supervisor": self.es_supervisor,
            "estado": self.estado,
            "fecha_ingreso": self.fecha_ingreso.isoformat(),
            "fecha_retorno_esperada": (
                self.fecha_retorno_esperada.isoformat()
                if self.fecha_retorno_esperada else ""
            ),
        }
        for emp in empresas:
            col = f"dist_{emp.replace(' ', '_')}"
            row[col] = self.distancias.get(emp, "")
        return row


# ---------------------------------------------------------------------------
# GuardManager
# ---------------------------------------------------------------------------


class GuardManager:
    """Registro dinámico de guardias de seguridad para el modelo de asignación.

    Gestiona el ciclo de vida de los guardias (alta, baja, reactivación),
    valida la consistencia de los datos y protege la factibilidad del
    modelo ILP antes de aprobar cada cambio de estado.

    Parameters
    ----------
    empresas : list[str], optional
        Lista de empresas. Por defecto usa ``EMPRESAS`` de
        ``models.asignacion``.
    puestos : dict[str, dict], optional
        Dotación mínima/máxima y requerimiento de supervisor por empresa.
        Por defecto usa ``PUESTOS`` de ``models.asignacion``.
    min_supervisores_global : int
        Número mínimo absoluto de supervisores activos en todo momento.
        Por defecto 1.

    Examples
    --------
    Uso básico:

    >>> gm = GuardManager.desde_asignacion()
    >>> gm.desactivar_guardia("G3", "Vacaciones anuales",
    ...                        nuevo_estado="vacaciones",
    ...                        fecha_retorno="2025-08-20")
    >>> activos = gm.ids_activos()   # lista para pasar al solver
    """

    # ------------------------------------------------------------------
    # Constructor y factory
    # ------------------------------------------------------------------

    def __init__(
        self,
        empresas: Optional[list[str]] = None,
        puestos: Optional[dict[str, dict]] = None,
        min_supervisores_global: int = 1,
    ) -> None:
        # Importación diferida para no crear dependencia dura en tests
        if empresas is None or puestos is None:
            try:
                from models.asignacion import (
                    EMPRESAS as _E,
                    PUESTOS as _P,
                )
                empresas = empresas or list(_E)
                puestos = puestos or dict(_P)
            except ImportError:
                empresas = empresas or ["Noramco", "ITI Chile", "Oleoducto", "Indama"]
                puestos = puestos or {
                    "Noramco":   {"min": 3, "max": 5, "requiere_supervisor": True},
                    "ITI Chile": {"min": 2, "max": 4, "requiere_supervisor": False},
                    "Oleoducto": {"min": 3, "max": 5, "requiere_supervisor": True},
                    "Indama":    {"min": 2, "max": 4, "requiere_supervisor": False},
                }

        self._empresas: list[str] = list(empresas)
        self._puestos: dict[str, dict] = dict(puestos)
        self._min_sup_global: int = min_supervisores_global
        self._guardias: dict[str, Guardia] = {}

    @classmethod
    def desde_asignacion(cls) -> "GuardManager":
        """Crea un GuardManager precargado con los 14 guardias de Bursan.

        Requiere que ``models.asignacion`` sea importable.

        Returns
        -------
        GuardManager
            Con todos los guardias ya registrados en estado 'activo'.
        """
        from models.asignacion import DISTANCIAS, GUARDIAS, SUPERVISORES

        gm = cls()
        for gid in GUARDIAS:
            gm.guardar_guardia(
                id=gid,
                nombre=f"Guardia {gid}",
                es_supervisor=gid in SUPERVISORES,
                distancias=dict(DISTANCIAS[gid]),
            )
        return gm

    # ------------------------------------------------------------------
    # Propiedades derivadas de la configuración
    # ------------------------------------------------------------------

    @property
    def total_min_dotacion(self) -> int:
        """Suma de dotaciones mínimas de todas las empresas."""
        return sum(p["min"] for p in self._puestos.values())

    @property
    def total_max_dotacion(self) -> int:
        """Suma de dotaciones máximas de todas las empresas."""
        return sum(p["max"] for p in self._puestos.values())

    @property
    def empresas_con_supervisor(self) -> list[str]:
        """Empresas que requieren al menos un supervisor asignado."""
        return [e for e, p in self._puestos.items() if p.get("requiere_supervisor")]

    @property
    def min_supervisores_requeridos(self) -> int:
        """Número mínimo de supervisores activos para que el ILP sea factible."""
        # Necesitamos 1 supervisor por empresa que lo requiere (R4),
        # y el mínimo global configurado.
        return max(self._min_sup_global, len(self.empresas_con_supervisor))

    # ------------------------------------------------------------------
    # Validaciones internas
    # ------------------------------------------------------------------

    def _validar_estado(self, estado: str) -> None:
        if estado not in ESTADOS_VALIDOS:
            raise EstadoInvalido(
                f"Estado '{estado}' no reconocido. "
                f"Estados válidos: {sorted(ESTADOS_VALIDOS)}."
            )

    def _validar_distancias(self, id_guardia: str, distancias: dict[str, float]) -> None:
        """Verifica que las distancias cubran todas las empresas y sean > 0."""
        faltantes = set(self._empresas) - set(distancias.keys())
        if faltantes:
            raise ValueError(
                f"Guardia '{id_guardia}': faltan distancias a "
                f"{sorted(faltantes)}. "
                f"Se requieren distancias a: {self._empresas}."
            )
        no_positivas = {e: d for e, d in distancias.items() if d <= 0}
        if no_positivas:
            raise ValueError(
                f"Guardia '{id_guardia}': distancias no positivas detectadas: "
                f"{no_positivas}. Todas las distancias deben ser > 0 km."
            )

    def _verificar_factibilidad_baja(self, id_a_bajar: str) -> None:
        """Bloquea la baja si el modelo ILP quedaría infactible.

        Comprobaciones (necesarias, no suficientes — el ILP es la autoridad):

        1. Guardias activos restantes ≥ dotación mínima total (R2).
        2. Supervisores activos restantes ≥ número de empresas que los
           requieren (R4 — cada empresa necesita al menos uno).

        Raises
        ------
        OperacionBloqueada
            Con un mensaje que explica exactamente qué restricción fallaría
            y qué acción correctiva tomar.
        """
        activos_restantes = [
            g for g in self._guardias.values()
            if g.activo and g.id != id_a_bajar
        ]

        # R2 — dotación mínima
        if len(activos_restantes) < self.total_min_dotacion:
            raise OperacionBloqueada(
                f"No se puede desactivar '{id_a_bajar}': quedarían "
                f"{len(activos_restantes)} guardias activos, pero la dotación "
                f"mínima total es {self.total_min_dotacion}. "
                "Agrega o activa otro guardia antes de realizar esta baja."
            )

        # R4 — supervisores por empresa
        sups_restantes = [g for g in activos_restantes if g.es_supervisor]
        n_req = self.min_supervisores_requeridos
        if len(sups_restantes) < n_req:
            raise OperacionBloqueada(
                f"No se puede desactivar '{id_a_bajar}': quedarían "
                f"{len(sups_restantes)} supervisor(es) activo(s), pero se "
                f"necesitan al menos {n_req} (una por cada empresa que requiere "
                f"supervisor: {self.empresas_con_supervisor}). "
                "Agrega o activa otro supervisor antes de esta baja."
            )

    def _registrar(
        self,
        guardia: Guardia,
        accion: str,
        estado_anterior: str,
        estado_nuevo: str,
        motivo: str = "",
        fecha_retorno: Optional[date] = None,
    ) -> None:
        guardia.historial.append(EntradaHistorial(
            timestamp=datetime.now(),
            accion=accion,
            estado_anterior=estado_anterior,
            estado_nuevo=estado_nuevo,
            motivo=motivo,
            fecha_retorno=fecha_retorno,
        ))

    # ------------------------------------------------------------------
    # Ciclo de vida de guardias
    # ------------------------------------------------------------------

    def guardar_guardia(
        self,
        id: str,
        nombre: str,
        es_supervisor: bool,
        distancias: dict[str, float],
        estado: EstadoGuardia = "activo",
        fecha_ingreso: Optional[date] = None,
    ) -> Guardia:
        """Registra un nuevo guardia en el sistema.

        Parameters
        ----------
        id : str
            Identificador único (ej. ``'G15'``). Falla si ya existe.
        nombre : str
            Nombre completo del guardia.
        es_supervisor : bool
            True si tiene certificación de supervisor.
        distancias : dict[str, float]
            Distancias en km a **todas** las empresas. Deben ser > 0.
        estado : EstadoGuardia
            Estado inicial. Por defecto ``'activo'``.
        fecha_ingreso : date, optional
            Fecha de ingreso al sistema. Por defecto hoy.

        Returns
        -------
        Guardia
            El objeto guardia creado.

        Raises
        ------
        ValueError
            Si el ID ya existe, el estado es inválido o las distancias
            son incorrectas (faltantes o no positivas).
        """
        id = (id or "").strip()
        if not id:
            raise ValueError("El ID del guardia no puede ser vacío o solo espacios.")
        if id in self._guardias:
            raise ValueError(
                f"El ID '{id}' ya existe en el registro. "
                f"IDs actuales: {sorted(self._guardias.keys())}."
            )

        self._validar_estado(estado)
        self._validar_distancias(id, distancias)

        guardia = Guardia(
            id=id,
            nombre=nombre.strip(),
            es_supervisor=bool(es_supervisor),
            distancias={e: float(d) for e, d in distancias.items()},
            estado=estado,
            fecha_ingreso=fecha_ingreso or date.today(),
        )
        self._registrar(
            guardia,
            accion="alta",
            estado_anterior="",
            estado_nuevo=estado,
            motivo="Alta inicial en el sistema.",
        )
        self._guardias[id] = guardia
        return guardia

    def desactivar_guardia(
        self,
        id: str,
        motivo: str,
        nuevo_estado: EstadoGuardia = "fuera_de_servicio",
        fecha_retorno: Optional[date | str] = None,
    ) -> None:
        """Cambia el estado de un guardia activo a un estado inactivo.

        Comprueba que la baja no haga infactible el modelo ILP antes de
        aplicar el cambio.

        Parameters
        ----------
        id : str
            ID del guardia a desactivar.
        motivo : str
            Razón de la baja (texto libre).
        nuevo_estado : EstadoGuardia
            Estado destino; no puede ser ``'activo'``.
            Por defecto ``'fuera_de_servicio'``.
        fecha_retorno : date | str | None
            Fecha estimada de regreso (para ``'vacaciones'`` o ``'licencia'``).
            Acepta ``date`` o str ISO-8601 (``'2025-08-15'``).

        Raises
        ------
        GuardiaNoEncontrado
        EstadoInvalido
            Si ``nuevo_estado`` es inválido o igual a ``'activo'``.
        ValueError
            Si el guardia ya está inactivo.
        OperacionBloqueada
            Si la baja violaría dotación mínima o cobertura de supervisores.
        """
        if id not in self._guardias:
            raise GuardiaNoEncontrado(
                f"Guardia '{id}' no encontrado. "
                f"IDs registrados: {sorted(self._guardias.keys())}."
            )
        self._validar_estado(nuevo_estado)
        if nuevo_estado == "activo":
            raise EstadoInvalido(
                "Para reactivar un guardia use reactivar_guardia(), "
                "no desactivar_guardia()."
            )

        guardia = self._guardias[id]
        if not guardia.activo:
            raise ValueError(
                f"El guardia '{id}' ya está en estado '{guardia.estado}'. "
                "Solo se pueden desactivar guardias con estado 'activo'."
            )

        self._verificar_factibilidad_baja(id)

        # Parsear fecha_retorno
        fecha_ret: Optional[date] = None
        if fecha_retorno is not None:
            fecha_ret = (
                date.fromisoformat(fecha_retorno)
                if isinstance(fecha_retorno, str)
                else fecha_retorno
            )

        estado_anterior = guardia.estado
        guardia.estado = nuevo_estado
        guardia.fecha_retorno_esperada = fecha_ret

        self._registrar(
            guardia,
            accion="baja",
            estado_anterior=estado_anterior,
            estado_nuevo=nuevo_estado,
            motivo=motivo,
            fecha_retorno=fecha_ret,
        )

    def reactivar_guardia(
        self,
        id: str,
        motivo: str = "Reincorporacion al servicio",
    ) -> None:
        """Reactiva un guardia, cambiando su estado a ``'activo'``.

        Parameters
        ----------
        id : str
        motivo : str
            Razón de la reactivación.

        Raises
        ------
        GuardiaNoEncontrado
        ValueError
            Si el guardia ya está activo.
        OperacionBloqueada
            Si el guardia fue ``'despedido'``: no se puede reactivar.
            Para volver a incluirlo se debe registrar como guardia nuevo.
        """
        if id not in self._guardias:
            raise GuardiaNoEncontrado(f"Guardia '{id}' no encontrado.")

        guardia = self._guardias[id]

        if guardia.activo:
            raise ValueError(
                f"El guardia '{id}' ya está en estado 'activo'."
            )
        if guardia.estado == "despedido":
            raise OperacionBloqueada(
                f"No se puede reactivar '{id}': fue despedido. "
                "Un guardia despedido no puede volver al servicio desde este "
                "sistema. Si corresponde, regístrelo como un nuevo guardia."
            )

        estado_anterior = guardia.estado
        guardia.estado = "activo"
        guardia.fecha_retorno_esperada = None

        self._registrar(
            guardia,
            accion="reactivacion",
            estado_anterior=estado_anterior,
            estado_nuevo="activo",
            motivo=motivo,
        )

    # ------------------------------------------------------------------
    # Consultas
    # ------------------------------------------------------------------

    def get_guardias_activos(self) -> list[Guardia]:
        """Guardias con estado ``'activo'``, ordenados por ID."""
        return sorted(
            (g for g in self._guardias.values() if g.activo),
            key=lambda g: g.id,
        )

    def get_todos_guardias(self) -> list[Guardia]:
        """Todos los guardias registrados, ordenados por ID."""
        return sorted(self._guardias.values(), key=lambda g: g.id)

    def get_guardia(self, id: str) -> Guardia:
        """Retorna un guardia por ID.

        Raises
        ------
        GuardiaNoEncontrado
        """
        if id not in self._guardias:
            raise GuardiaNoEncontrado(f"Guardia '{id}' no encontrado.")
        return self._guardias[id]

    def ids_activos(self) -> list[str]:
        """IDs de guardias activos (compatible con ``resolver(guardias_activos=...)``).

        Returns
        -------
        list[str]
            Lista ordenada de IDs.
        """
        return [g.id for g in self.get_guardias_activos()]

    def n_activos(self) -> int:
        """Número de guardias con estado 'activo'."""
        return len(self.get_guardias_activos())

    def n_supervisores_activos(self) -> int:
        """Número de supervisores con estado 'activo'."""
        return sum(1 for g in self.get_guardias_activos() if g.es_supervisor)

    def distancias_activos(self) -> dict[str, dict[str, float]]:
        """Distancias de los guardias activos en el formato de ``DISTANCIAS``.

        Compatible para pasar directamente a cualquier solver o heurística.

        Returns
        -------
        dict[str, dict[str, float]]
            ``{guardia_id: {empresa: km, ...}}``
        """
        return {g.id: dict(g.distancias) for g in self.get_guardias_activos()}

    def es_factible(self) -> bool:
        """True si los activos actuales satisfacen las condiciones necesarias."""
        return (
            self.n_activos() >= self.total_min_dotacion
            and self.n_supervisores_activos() >= self.min_supervisores_requeridos
        )

    def resumen(self) -> str:
        """Resumen legible del estado del registro."""
        activos = self.get_guardias_activos()
        todos = self.get_todos_guardias()
        sups = [g for g in activos if g.es_supervisor]

        por_estado: dict[str, int] = {}
        for g in todos:
            por_estado[g.estado] = por_estado.get(g.estado, 0) + 1

        factible = self.es_factible()
        return (
            f"GuardManager — {len(todos)} guardias registrados\n"
            f"  Activos     : {len(activos)}  ({len(sups)} supervisores)\n"
            f"  Dot. minima : {self.total_min_dotacion}  "
            f"Sup. requeridos: {self.min_supervisores_requeridos}\n"
            f"  Factible    : {'SI' if factible else 'NO - revisar dotacion'}\n"
            f"  Por estado  : {por_estado}"
        )

    # ------------------------------------------------------------------
    # Integración con el solver
    # ------------------------------------------------------------------

    def resolver_asignacion(self, **kwargs):
        """Ejecuta el modelo ILP con los guardias activos actuales.

        Delega a ``models.asignacion.resolver`` pasando ``guardias_activos``
        automáticamente. El resto de parámetros se pasan tal cual.

        Parameters
        ----------
        **kwargs
            Argumentos de :func:`models.asignacion.resolver`
            (``modo_objetivo``, ``alpha``, ``beta``, ``delta_max``, etc.).

        Returns
        -------
        ResultadoAsignacion

        Raises
        ------
        ImportError
            Si ``models.asignacion`` no está disponible.
        """
        from models.asignacion import resolver
        return resolver(guardias_activos=self.ids_activos(), **kwargs)

    # ------------------------------------------------------------------
    # DataFrames
    # ------------------------------------------------------------------

    def to_dataframe(self) -> pd.DataFrame:
        """Estado actual de todos los guardias como DataFrame."""
        if not self._guardias:
            return pd.DataFrame()
        return pd.DataFrame(
            [g.to_dict(self._empresas) for g in self.get_todos_guardias()]
        )

    def historial_dataframe(self) -> pd.DataFrame:
        """Historial completo de todos los guardias, ordenado por timestamp."""
        rows = [
            entry.to_dict(g.id)
            for g in self.get_todos_guardias()
            for entry in g.historial
        ]
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)

    # ------------------------------------------------------------------
    # I/O CSV
    # ------------------------------------------------------------------

    def exportar_csv(self, path: str | Path) -> None:
        """Exporta el registro completo a dos archivos CSV.

        Archivos generados:

        * ``path``                       → estado actual de cada guardia.
        * ``{path.stem}_historial.csv``  → historial completo de cambios.

        Ambos archivos son suficientes para restaurar el estado exacto
        mediante :meth:`importar_csv`.

        Parameters
        ----------
        path : str | Path
            Ruta destino del CSV de guardias.

        Raises
        ------
        OSError
            Si no se puede escribir en la ubicación indicada.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        df = self.to_dataframe()
        if not df.empty:
            df.to_csv(path, index=False, encoding="utf-8")
        else:
            path.write_text("", encoding="utf-8")

        hist_path = path.with_name(f"{path.stem}_historial.csv")
        hist_df = self.historial_dataframe()
        if not hist_df.empty:
            hist_df.to_csv(hist_path, index=False, encoding="utf-8")
        else:
            hist_path.write_text("", encoding="utf-8")

    def importar_csv(self, path: str | Path) -> None:
        """Restaura el registro desde un CSV exportado con :meth:`exportar_csv`.

        Reemplaza cualquier estado en memoria. Si existe el archivo
        ``{stem}_historial.csv`` en la misma carpeta, también restaura
        el historial de cambios.

        Parameters
        ----------
        path : str | Path

        Raises
        ------
        FileNotFoundError
        ValueError
            Si el CSV no tiene las columnas obligatorias.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {path}")

        content = path.read_text(encoding="utf-8").strip()
        if not content:
            self._guardias = {}
            return

        df = pd.read_csv(path, encoding="utf-8")
        if df.empty:
            self._guardias = {}
            return

        required = {"id", "nombre", "es_supervisor", "estado"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"Columnas obligatorias faltantes en '{path.name}': {missing}."
            )

        # Mapeo columnas dist_X → empresa canonical
        dist_col_to_empresa: dict[str, str] = {}
        for col in df.columns:
            if not col.startswith("dist_"):
                continue
            col_empresa = col[len("dist_"):].replace("_", " ")
            for emp in self._empresas:
                if emp.replace(" ", "_") == col_empresa.replace(" ", "_"):
                    dist_col_to_empresa[col] = emp
                    break

        self._guardias = {}

        for _, row in df.iterrows():
            distancias = {
                empresa: float(row[col])
                for col, empresa in dist_col_to_empresa.items()
                if pd.notna(row.get(col))
            }

            fecha_ingreso = date.today()
            raw_fi = str(row.get("fecha_ingreso", "")).strip()
            if raw_fi:
                try:
                    fecha_ingreso = date.fromisoformat(raw_fi)
                except ValueError:
                    pass

            fecha_ret: Optional[date] = None
            raw_fr = str(row.get("fecha_retorno_esperada", "")).strip()
            if raw_fr:
                try:
                    fecha_ret = date.fromisoformat(raw_fr)
                except ValueError:
                    pass

            g = Guardia(
                id=str(row["id"]),
                nombre=str(row["nombre"]),
                es_supervisor=bool(row["es_supervisor"]),
                distancias=distancias,
                estado=str(row["estado"]),
                fecha_ingreso=fecha_ingreso,
                fecha_retorno_esperada=fecha_ret,
            )
            self._guardias[g.id] = g

        # Restaurar historial
        hist_path = path.with_name(f"{path.stem}_historial.csv")
        if hist_path.exists():
            hist_raw = hist_path.read_text(encoding="utf-8").strip()
            if hist_raw:
                hist_df = pd.read_csv(hist_path, encoding="utf-8")
                for _, row in hist_df.iterrows():
                    gid = str(row.get("guardia_id", ""))
                    if gid not in self._guardias:
                        continue

                    fecha_ret_h: Optional[date] = None
                    raw = str(row.get("fecha_retorno", "")).strip()
                    if raw:
                        try:
                            fecha_ret_h = date.fromisoformat(raw)
                        except ValueError:
                            pass

                    try:
                        ts = datetime.fromisoformat(str(row["timestamp"]))
                    except (ValueError, KeyError):
                        ts = datetime.now()

                    self._guardias[gid].historial.append(EntradaHistorial(
                        timestamp=ts,
                        accion=str(row.get("accion", "")),
                        estado_anterior=str(row.get("estado_anterior", "")),
                        estado_nuevo=str(row.get("estado_nuevo", "")),
                        motivo=str(row.get("motivo", "")),
                        fecha_retorno=fecha_ret_h,
                    ))


# ---------------------------------------------------------------------------
# Verificación rápida (python -m app.components.guard_manager)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from models.asignacion import DISTANCIAS, GUARDIAS, SUPERVISORES

    SEP = "=" * 58

    print(SEP)
    print("Verificacion rapida — GuardManager")
    print(SEP)

    # ------------------------------------------------------------------ #
    # T1. Cargar instancia Bursan completa                               #
    # ------------------------------------------------------------------ #
    print("\n[T1] Cargar los 14 guardias de Bursan")
    gm = GuardManager.desde_asignacion()
    assert gm.n_activos() == 14
    assert gm.n_supervisores_activos() == 2
    assert gm.es_factible()
    print(gm.resumen())
    print("  PASA")

    # ------------------------------------------------------------------ #
    # T2. Alta de un guardia nuevo                                       #
    # ------------------------------------------------------------------ #
    print("\n[T2] Alta guardia G15 (nuevo)")
    gm.guardar_guardia(
        id="G15",
        nombre="Pedro Soto",
        es_supervisor=False,
        distancias={e: 10.0 for e in gm._empresas},
        estado="activo",
    )
    assert gm.n_activos() == 15
    assert "G15" in [g.id for g in gm.get_guardias_activos()]
    print(f"  Activos ahora: {gm.n_activos()}")
    print("  PASA")

    # ------------------------------------------------------------------ #
    # T3. Baja temporal (vacaciones)                                     #
    # ------------------------------------------------------------------ #
    print("\n[T3] Baja temporal G3 (vacaciones)")
    gm.desactivar_guardia(
        "G3", motivo="Vacaciones de verano",
        nuevo_estado="vacaciones", fecha_retorno="2025-08-31"
    )
    g3 = gm.get_guardia("G3")
    assert g3.estado == "vacaciones"
    assert g3.fecha_retorno_esperada is not None
    assert "G3" not in gm.ids_activos()
    print(f"  G3 estado: {g3.estado}  retorno: {g3.fecha_retorno_esperada}")
    print("  PASA")

    # ------------------------------------------------------------------ #
    # T4. Reactivacion                                                   #
    # ------------------------------------------------------------------ #
    print("\n[T4] Reactivar G3")
    gm.reactivar_guardia("G3", motivo="Regreso de vacaciones")
    assert gm.get_guardia("G3").activo
    assert "G3" in gm.ids_activos()
    print("  PASA")

    # ------------------------------------------------------------------ #
    # T5. Baja bloqueada por dotacion minima                             #
    # ------------------------------------------------------------------ #
    print("\n[T5] Bloqueo por dotacion minima")
    # Desactivar hasta dejar exactamente total_min activos, luego intentar una mas
    ids_a_bajar = [g for g in gm.ids_activos()
                   if not gm.get_guardia(g).es_supervisor][: gm.n_activos() - gm.total_min_dotacion]
    for g_id in ids_a_bajar:
        gm.desactivar_guardia(g_id, motivo="Test baja masiva")
    assert gm.n_activos() == gm.total_min_dotacion
    candidato = next(g for g in gm.ids_activos() if not gm.get_guardia(g).es_supervisor)
    try:
        gm.desactivar_guardia(candidato, motivo="Baja que debe bloquearse")
        assert False, "Debia lanzar OperacionBloqueada"
    except OperacionBloqueada as e:
        print(f"  Bloqueada correctamente: {e}")
    print("  PASA")

    # ------------------------------------------------------------------ #
    # T6. Baja bloqueada por supervisores                               #
    # ------------------------------------------------------------------ #
    print("\n[T6] Bloqueo por supervisor")
    # Restablecer: reactivar todos para este test
    gm2 = GuardManager.desde_asignacion()
    # Con solo G1 y G6 como supervisores, no se puede bajar ninguno
    try:
        gm2.desactivar_guardia("G1", motivo="Test supervisor")
        assert False, "Debia lanzar OperacionBloqueada"
    except OperacionBloqueada as e:
        print(f"  Bloqueada correctamente: {e}")
    print("  PASA")

    # ------------------------------------------------------------------ #
    # T7. Despido (no reactivable)                                      #
    # ------------------------------------------------------------------ #
    print("\n[T7] Despido y bloqueo de reactivacion")
    gm2.desactivar_guardia("G14", motivo="Conducta inapropiada", nuevo_estado="despedido")
    try:
        gm2.reactivar_guardia("G14")
        assert False, "Debia lanzar OperacionBloqueada"
    except OperacionBloqueada as e:
        print(f"  Bloqueada correctamente: {e}")
    print("  PASA")

    # ------------------------------------------------------------------ #
    # T8. Historial registrado correctamente                            #
    # ------------------------------------------------------------------ #
    print("\n[T8] Historial de G3")
    g3_hist = gm.get_guardia("G3").historial
    assert len(g3_hist) == 3  # alta, baja, reactivacion
    assert g3_hist[0].accion == "alta"
    assert g3_hist[1].accion == "baja"
    assert g3_hist[2].accion == "reactivacion"
    for h in g3_hist:
        print(f"    [{h.timestamp.strftime('%H:%M:%S')}] {h.accion}: "
              f"{h.estado_anterior or 'INICIO'} -> {h.estado_nuevo}  | {h.motivo}")
    print("  PASA")

    # ------------------------------------------------------------------ #
    # T9. Roundtrip CSV                                                  #
    # ------------------------------------------------------------------ #
    import tempfile, os
    print("\n[T9] Roundtrip CSV")
    gm3 = GuardManager.desde_asignacion()
    gm3.desactivar_guardia("G5", motivo="Licencia medica",
                            nuevo_estado="licencia", fecha_retorno="2025-09-01")
    gm3.reactivar_guardia("G5", motivo="Alta medica")

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "guardias.csv")
        gm3.exportar_csv(csv_path)

        gm4 = GuardManager()
        gm4.importar_csv(csv_path)

        assert gm4.n_activos() == gm3.n_activos()
        assert gm4.ids_activos() == gm3.ids_activos()

        g5_orig = gm3.get_guardia("G5")
        g5_rest = gm4.get_guardia("G5")
        assert g5_rest.estado == g5_orig.estado
        assert len(g5_rest.historial) == len(g5_orig.historial)
    print(f"  Exportado y restaurado: {gm3.n_activos()} activos, historial OK")
    print("  PASA")

    print(f"\n{SEP}")
    print("Todas las pruebas pasaron.")
    print(SEP)
