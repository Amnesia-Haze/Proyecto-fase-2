"""instances.py — Modelo de datos, I/O y visualización para CVRP.

El nodo 0 es siempre el depósito (demands[0] == 0).
No contiene heurísticas ni solvers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# ---------------------------------------------------------------------------
# Modelo de datos
# ---------------------------------------------------------------------------

@dataclass
class VRPInstance:
    """Instancia de CVRP (Capacitated Vehicle Routing Problem).

    Parameters
    ----------
    coords : np.ndarray, shape (N, 2)
        Coordenadas de todos los nodos. ``coords[0]`` es el depósito.
    demands : np.ndarray, shape (N,)
        Demanda de cada nodo. ``demands[0]`` debe ser 0 (depósito).
    capacity : float
        Capacidad Q de cada vehículo.
    name : str
        Nombre descriptivo de la instancia.
    metadata : dict
        Información adicional libre (autor, fuente, fecha, etc.).
    """

    coords: np.ndarray
    demands: np.ndarray
    capacity: float
    name: str = "Instancia CVRP"
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.coords = np.asarray(self.coords, dtype=float)
        self.demands = np.asarray(self.demands, dtype=float)
        self._validate()

    # ------------------------------------------------------------------
    # Validación
    # ------------------------------------------------------------------

    def _validate(self) -> None:
        if self.coords.ndim != 2 or self.coords.shape[1] != 2:
            raise ValueError(
                f"'coords' debe tener shape (N, 2), se obtuvo {self.coords.shape}."
            )
        n = len(self.coords)
        if len(self.demands) != n:
            raise ValueError(
                f"'demands' debe tener longitud {n} (igual que coords), "
                f"se obtuvo {len(self.demands)}."
            )
        if n < 2:
            raise ValueError("La instancia debe tener al menos 1 cliente (N >= 2).")
        if self.demands[0] != 0.0:
            raise ValueError(
                f"La demanda del depósito (nodo 0) debe ser 0, "
                f"se obtuvo {self.demands[0]}."
            )
        neg_nodes = np.where(self.demands < 0)[0]
        if len(neg_nodes) > 0:
            raise ValueError(
                f"Se encontraron demandas negativas en los nodos: {neg_nodes.tolist()}."
            )
        if self.capacity <= 0:
            raise ValueError(
                f"La capacidad debe ser positiva, se obtuvo {self.capacity}."
            )
        over_nodes = np.where(self.demands > self.capacity)[0]
        if len(over_nodes) > 0:
            raise ValueError(
                f"Los nodos {over_nodes.tolist()} tienen demanda mayor que "
                f"la capacidad Q={self.capacity}."
            )

    # ------------------------------------------------------------------
    # Propiedades
    # ------------------------------------------------------------------

    @property
    def n_nodes(self) -> int:
        """Número total de nodos (depósito + clientes)."""
        return len(self.coords)

    @property
    def n_customers(self) -> int:
        """Número de clientes (excluye el depósito)."""
        return len(self.coords) - 1

    # ------------------------------------------------------------------
    # Métodos
    # ------------------------------------------------------------------

    def distance_matrix(self) -> np.ndarray:
        """Calcula la matriz de distancias euclidianas entre todos los nodos.

        Returns
        -------
        np.ndarray, shape (N, N)
            ``D[i, j]`` es la distancia euclidiana entre el nodo i y el nodo j.
        """
        diff = self.coords[:, np.newaxis, :] - self.coords[np.newaxis, :, :]
        return np.sqrt((diff ** 2).sum(axis=2))

    def summary(self) -> dict:
        """Resumen estadístico de la instancia."""
        customer_demands = self.demands[1:]
        return {
            "Nombre": self.name,
            "Nodos totales (N)": self.n_nodes,
            "Clientes": self.n_customers,
            "Capacidad (Q)": self.capacity,
            "Demanda total": float(customer_demands.sum()),
            "Demanda promedio": round(float(customer_demands.mean()), 2),
            "Demanda mínima": float(customer_demands.min()),
            "Demanda máxima": float(customer_demands.max()),
            "Vehículos mínimos requeridos (cota inferior)": int(
                np.ceil(customer_demands.sum() / self.capacity)
            ),
        }


# ---------------------------------------------------------------------------
# Generación de instancias
# ---------------------------------------------------------------------------

def generate_cvrp_instance(
    n_customers: int,
    capacity: float,
    coord_range: tuple[float, float] = (0.0, 100.0),
    demand_range: tuple[float, float] = (1.0, 10.0),
    depot_coords: tuple[float, float] = (50.0, 50.0),
    seed: int = 42,
    name: Optional[str] = None,
) -> VRPInstance:
    """Genera una instancia CVRP aleatoria con semilla reproducible.

    Parameters
    ----------
    n_customers : int
        Número de clientes (nodos 1 … n_customers).
    capacity : float
        Capacidad Q de cada vehículo.
    coord_range : tuple[float, float]
        Rango (min, max) para coordenadas x e y de los clientes.
    demand_range : tuple[float, float]
        Rango (min, max) para la demanda por cliente (redondeada a 1 decimal).
        Si d_max > capacity, se recorta a capacity automáticamente.
    depot_coords : tuple[float, float]
        Coordenadas fijas del depósito (nodo 0).
    seed : int
        Semilla para reproducibilidad (usa ``numpy.random.default_rng``).
    name : str, optional
        Nombre de la instancia. Si es None, se genera automáticamente.

    Returns
    -------
    VRPInstance

    Raises
    ------
    ValueError
        Si n_customers < 1, capacity <= 0 o demand_range es inválido.
    """
    if n_customers < 1:
        raise ValueError("n_customers debe ser >= 1.")
    if capacity <= 0:
        raise ValueError("capacity debe ser positiva.")
    d_min, d_max = demand_range
    if d_min <= 0:
        raise ValueError("El mínimo de demand_range debe ser positivo.")
    if d_min > capacity:
        raise ValueError(
            f"El mínimo de demand_range ({d_min}) supera la capacidad ({capacity})."
        )
    d_max = min(d_max, capacity)

    rng = np.random.default_rng(seed)

    customer_coords = rng.uniform(coord_range[0], coord_range[1], size=(n_customers, 2))
    customer_demands = np.round(rng.uniform(d_min, d_max, size=n_customers), 1)

    coords = np.vstack([[depot_coords], customer_coords])
    demands = np.concatenate([[0.0], customer_demands])

    instance_name = name or f"CVRP-n{n_customers}-Q{int(capacity)}-seed{seed}"

    return VRPInstance(
        coords=coords,
        demands=demands,
        capacity=capacity,
        name=instance_name,
        metadata={
            "seed": seed,
            "coord_range": coord_range,
            "demand_range": (d_min, d_max),
            "generated_by": "generate_cvrp_instance",
        },
    )


# ---------------------------------------------------------------------------
# I/O CSV
# ---------------------------------------------------------------------------

def read_instance_csv(filepath: str | Path) -> VRPInstance:
    """Lee una instancia CVRP desde un archivo CSV.

    Formato esperado — columnas obligatorias:

    ======== ======= ==========================================
    node     int     índice de nodo; 0 = depósito
    x        float   coordenada x
    y        float   coordenada y
    demand   float   demanda (0 para el depósito)
    capacity float   capacidad Q (mismo valor en todas las filas)
    ======== ======= ==========================================

    Columna opcional: ``name`` (str) — nombre de la instancia.

    Parameters
    ----------
    filepath : str | Path

    Returns
    -------
    VRPInstance

    Raises
    ------
    FileNotFoundError
        Si el archivo no existe.
    ValueError
        Si faltan columnas obligatorias o los datos no son válidos.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {path}")

    df = pd.read_csv(path)

    required = {"node", "x", "y", "demand", "capacity"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"El CSV debe contener las columnas: {required}. Faltan: {missing}."
        )

    df = df.sort_values("node").reset_index(drop=True)

    if int(df["node"].iloc[0]) != 0:
        raise ValueError("El CSV debe incluir el nodo 0 (depósito).")
    if df["node"].duplicated().any():
        raise ValueError("Los índices de nodo deben ser únicos.")

    coords = df[["x", "y"]].to_numpy(dtype=float)
    demands = df["demand"].to_numpy(dtype=float)
    capacity = float(df["capacity"].iloc[0])
    instance_name = str(df["name"].iloc[0]) if "name" in df.columns else path.stem

    return VRPInstance(
        coords=coords,
        demands=demands,
        capacity=capacity,
        name=instance_name,
        metadata={"source_file": str(path)},
    )


def write_instance_csv(instance: VRPInstance, filepath: str | Path) -> None:
    """Guarda una instancia CVRP en un archivo CSV compatible con :func:`read_instance_csv`.

    Parameters
    ----------
    instance : VRPInstance
    filepath : str | Path
        Ruta de destino. Los directorios intermedios se crean automáticamente.

    Raises
    ------
    OSError
        Si no se puede escribir el archivo.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    n = instance.n_nodes
    df = pd.DataFrame({
        "node": np.arange(n, dtype=int),
        "x": instance.coords[:, 0],
        "y": instance.coords[:, 1],
        "demand": instance.demands,
        "capacity": instance.capacity,
        "name": instance.name,
    })
    df.to_csv(path, index=False, float_format="%.4f")


# ---------------------------------------------------------------------------
# Visualización
# ---------------------------------------------------------------------------

def plot_instance(
    instance: VRPInstance,
    title: Optional[str] = None,
) -> go.Figure:
    """Genera un gráfico Plotly de la instancia CVRP (solo nodos, sin rutas).

    El depósito aparece como cuadrado negro. Los clientes se colorean por
    demanda (escala de azules) con la etiqueta ``Ci / d=<demanda>``.

    Parameters
    ----------
    instance : VRPInstance
    title : str, optional
        Título del gráfico. Por defecto usa ``instance.name``.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    fig = go.Figure()

    # Depósito
    fig.add_trace(go.Scatter(
        x=[instance.coords[0, 0]],
        y=[instance.coords[0, 1]],
        mode="markers+text",
        text=["Depósito"],
        textposition="top center",
        marker=dict(size=18, color="black", symbol="square"),
        name="Depósito",
        hovertemplate="Depósito<br>(%{x:.1f}, %{y:.1f})<extra></extra>",
    ))

    # Clientes
    cx = instance.coords[1:, 0]
    cy = instance.coords[1:, 1]
    labels = [
        f"C{i}<br>d={instance.demands[i]:.1f}"
        for i in range(1, instance.n_nodes)
    ]
    hover = [
        f"Nodo {i} | Demanda: {instance.demands[i]:.1f}<br>({instance.coords[i,0]:.1f}, {instance.coords[i,1]:.1f})"
        for i in range(1, instance.n_nodes)
    ]
    fig.add_trace(go.Scatter(
        x=cx,
        y=cy,
        mode="markers+text",
        text=labels,
        textposition="top center",
        customdata=hover,
        hovertemplate="%{customdata}<extra></extra>",
        marker=dict(
            size=11,
            color=instance.demands[1:],
            colorscale="Blues",
            showscale=True,
            colorbar=dict(title="Demanda"),
            line=dict(width=1, color="darkblue"),
        ),
        name="Clientes",
    ))

    fig.update_layout(
        title=title or instance.name,
        xaxis_title="X",
        yaxis_title="Y",
        legend_title="Nodos",
        height=520,
        hovermode="closest",
    )
    return fig


# ---------------------------------------------------------------------------
# Ejemplo mínimo de uso
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # 1. Generar instancia reproducible
    inst = generate_cvrp_instance(n_customers=10, capacity=50, seed=7)
    print("=== Resumen ===")
    for k, v in inst.summary().items():
        print(f"  {k}: {v}")

    # 2. Primeras filas de la matriz de distancias
    D = inst.distance_matrix()
    print(f"\nMatriz de distancias (4x4 primeros nodos):\n{D[:4, :4].round(2)}")

    # 3. Guardar y releer
    write_instance_csv(inst, "ejemplo_cvrp.csv")
    inst2 = read_instance_csv("ejemplo_cvrp.csv")
    assert inst2.n_customers == inst.n_customers
    print(f"\nInstancia re-leida: '{inst2.name}', clientes={inst2.n_customers}  OK")

    # 4. Visualizar (requiere navegador)
    fig = plot_instance(inst)
    fig.show()
