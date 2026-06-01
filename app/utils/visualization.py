"""visualization.py — Funciones de gráficos y exportación HTML para CVRP.

Contiene dos familias:
  • plot_routes / comparison_table / bar_comparison
        Trabajan con _LegacySolution (app.models.vrp_instance.VRPSolution).
        Usadas por las páginas multi-page (1_Instancias, 2_Resolver, 3_Comparar).

  • plot_cvrp_solution / build_html_report
        Trabajan con solution.VRPSolution (app.models.solution).
        Usadas por app.py.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from app.models import VRPSolution as _LegacySolution
from app.models.instances import VRPInstance

COLORS = px.colors.qualitative.Set2


# ---------------------------------------------------------------------------
# Familia legacy (multi-page app)
# ---------------------------------------------------------------------------

def plot_routes(solution: _LegacySolution) -> go.Figure:
    """Gráfico Plotly de rutas para _LegacySolution."""
    instance = solution.instance
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=[instance.coords[0, 0]],
        y=[instance.coords[0, 1]],
        mode="markers+text",
        text=["Depósito"],
        textposition="top center",
        marker=dict(size=16, color="black", symbol="square"),
        name="Depósito",
        hovertemplate="Depósito<br>(%{x:.1f}, %{y:.1f})<extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        x=instance.coords[1:, 0],
        y=instance.coords[1:, 1],
        mode="markers+text",
        text=[f"C{i}" for i in range(1, instance.n_nodes)],
        textposition="top center",
        marker=dict(size=9, color="steelblue"),
        name="Clientes",
        hovertemplate="Nodo %{text}<br>(%{x:.1f}, %{y:.1f})<extra></extra>",
    ))

    for idx, route in enumerate(solution.routes):
        if not route:
            continue
        color = COLORS[idx % len(COLORS)]
        nodes = [0] + route + [0]
        xs = [instance.coords[n, 0] for n in nodes]
        ys = [instance.coords[n, 1] for n in nodes]
        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode="lines+markers",
            line=dict(color=color, width=2),
            marker=dict(size=7),
            name=f"Ruta {idx + 1}",
        ))

    fig.update_layout(
        title=f"{solution.algorithm_name} — Distancia total: {solution.total_distance():.2f}",
        xaxis_title="X", yaxis_title="Y",
        legend_title="Leyenda", height=500,
    )
    return fig


def comparison_table(solutions: list[_LegacySolution]) -> pd.DataFrame:
    return pd.DataFrame([s.summary() for s in solutions])


def bar_comparison(solutions: list[_LegacySolution]) -> go.Figure:
    df = comparison_table(solutions)
    fig = px.bar(
        df, x="Algoritmo", y="Distancia total", color="Algoritmo",
        text="Distancia total", title="Comparación de distancia total por algoritmo",
    )
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    return fig


# ---------------------------------------------------------------------------
# Familia nueva (app.py + solution.VRPSolution)
# ---------------------------------------------------------------------------

def plot_cvrp_solution(sol: "NewSolution") -> go.Figure:  # type: ignore[name-defined]
    """Gráfico Plotly de rutas para solution.VRPSolution.

    Cada ruta recibe un color distinto. El hover muestra demanda y carga
    acumulada por ruta.
    """
    from app.models.solution import VRPSolution, route_load

    instance = sol.instance
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

    # Clientes (sin ruta, puntos de fondo)
    hover_clients = [
        f"C{i} | demanda={instance.demands[i]:.1f}<br>({instance.coords[i,0]:.1f}, {instance.coords[i,1]:.1f})"
        for i in range(1, instance.n_nodes)
    ]
    fig.add_trace(go.Scatter(
        x=instance.coords[1:, 0],
        y=instance.coords[1:, 1],
        mode="markers",
        marker=dict(size=8, color="lightsteelblue", line=dict(width=1, color="steelblue")),
        name="Clientes",
        customdata=hover_clients,
        hovertemplate="%{customdata}<extra></extra>",
        showlegend=True,
    ))

    # Rutas
    utils = sol.capacity_utilizations()
    for idx, route in enumerate(sol.routes):
        color = COLORS[idx % len(COLORS)]
        load = route_load(route, instance)
        pct = utils[idx]
        nodes = [0] + route + [0]
        xs = [instance.coords[n, 0] for n in nodes]
        ys = [instance.coords[n, 1] for n in nodes]
        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode="lines+markers",
            line=dict(color=color, width=2.2),
            marker=dict(size=7, color=color),
            name=f"Ruta {idx + 1} ({pct:.0f}%)",
            hovertemplate=f"Ruta {idx+1}<br>Carga: {load:.1f}/{instance.capacity} ({pct:.1f}%)<extra></extra>",
        ))

    feasible, errors = sol.feasibility_report()
    status = "✓ Factible" if feasible else "✗ Infactible"
    fig.update_layout(
        title=f"{sol.method}  |  Costo: {sol.total_cost():.2f}  |  {status}  |  Q={instance.capacity}",
        xaxis_title="X",
        yaxis_title="Y",
        legend_title="Rutas",
        height=520,
        hovermode="closest",
    )
    return fig


def build_html_report(sol: "NewSolution", fig: go.Figure) -> str:  # type: ignore[name-defined]
    """Genera un reporte HTML autocontenido con la solución CVRP.

    Incluye: resumen, factibilidad, tabla de rutas y gráfico Plotly embebido.
    """
    import datetime

    feasible, errors = sol.feasibility_report()
    feasibility_badge = (
        '<span style="color:green;font-weight:bold">✓ Factible</span>'
        if feasible
        else '<span style="color:red;font-weight:bold">✗ Infactible</span> — '
        + "; ".join(errors)
    )

    summary_df = pd.DataFrame([sol.summary()])
    summary_html = summary_df.to_html(index=False, border=0, classes="table")

    routes_df = sol.to_routes_dataframe()
    routes_html = routes_df.to_html(index=False, border=0, classes="table")

    chart_html = fig.to_html(full_html=False, include_plotlyjs="cdn")
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    return f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <title>Reporte CVRP — {sol.method}</title>
  <style>
    body  {{ font-family: Arial, sans-serif; margin: 40px; color: #222; }}
    h1    {{ color: #1a4e8a; }}
    h2    {{ color: #2c6fad; border-bottom: 1px solid #ccc; padding-bottom: 4px; }}
    .table {{ border-collapse: collapse; width: 100%; margin-bottom: 24px; }}
    .table th, .table td {{ border: 1px solid #ddd; padding: 7px 12px; text-align: left; }}
    .table th {{ background: #f0f4fa; }}
    .table tr:nth-child(even) {{ background: #f9f9f9; }}
    .meta {{ color: #555; font-size: 0.9em; margin-bottom: 24px; }}
  </style>
</head>
<body>
  <h1>Reporte CVRP</h1>
  <p class="meta">Generado: {now} | Instancia: {sol.instance.name} | Método: {sol.method}</p>

  <h2>Resumen de la solución</h2>
  {summary_html}

  <h2>Factibilidad</h2>
  <p>{feasibility_badge}</p>

  <h2>Detalle de rutas</h2>
  {routes_html}

  <h2>Visualización</h2>
  {chart_html}
</body>
</html>"""
