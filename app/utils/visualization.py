"""visualization.py — Gráficos y exportación HTML para CVRP.

Dos familias:
  · plot_routes / comparison_table / bar_comparison
        _LegacySolution — páginas multi-page (1_Instancias, 2_Resolver, 3_Comparar).
  · plot_cvrp_solution / plot_utilization_bars / compare_solutions_* / build_html_report
        solution.VRPSolution — app.py.
"""

from __future__ import annotations

import datetime
from typing import TYPE_CHECKING

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from app.models import VRPSolution as _LegacySolution
from app.models.instances import VRPInstance

if TYPE_CHECKING:
    from app.models.solution import VRPSolution as NewVRPSolution

COLORS = px.colors.qualitative.Safe   # paleta accesible (daltónica)


# ---------------------------------------------------------------------------
# Familia legacy — multi-page app
# ---------------------------------------------------------------------------

def plot_routes(solution: _LegacySolution) -> go.Figure:
    instance = solution.instance
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[instance.coords[0, 0]], y=[instance.coords[0, 1]],
        mode="markers+text", text=["Depósito"], textposition="top center",
        marker=dict(size=16, color="black", symbol="square"), name="Depósito",
        hovertemplate="Depósito<br>(%{x:.1f}, %{y:.1f})<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=instance.coords[1:, 0], y=instance.coords[1:, 1],
        mode="markers+text",
        text=[f"C{i}" for i in range(1, instance.n_nodes)],
        textposition="top center",
        marker=dict(size=9, color="steelblue"), name="Clientes",
        hovertemplate="Nodo %{text}<br>(%{x:.1f}, %{y:.1f})<extra></extra>",
    ))
    for idx, route in enumerate(solution.routes):
        if not route:
            continue
        color = COLORS[idx % len(COLORS)]
        nodes = [0] + route + [0]
        fig.add_trace(go.Scatter(
            x=[instance.coords[n, 0] for n in nodes],
            y=[instance.coords[n, 1] for n in nodes],
            mode="lines+markers", line=dict(color=color, width=2),
            marker=dict(size=7), name=f"Ruta {idx + 1}",
        ))
    fig.update_layout(
        title=f"{solution.algorithm_name} — Distancia total: {solution.total_distance():.2f}",
        xaxis_title="X", yaxis_title="Y", legend_title="Leyenda", height=500,
    )
    return fig


def comparison_table(solutions: list[_LegacySolution]) -> pd.DataFrame:
    return pd.DataFrame([s.summary() for s in solutions])


def bar_comparison(solutions: list[_LegacySolution]) -> go.Figure:
    df = comparison_table(solutions)
    fig = px.bar(df, x="Algoritmo", y="Distancia total", color="Algoritmo",
                 text="Distancia total",
                 title="Comparación de distancia total por algoritmo")
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    return fig


# ---------------------------------------------------------------------------
# Familia nueva — app.py / solution.VRPSolution
# ---------------------------------------------------------------------------

def plot_cvrp_solution(sol: "NewVRPSolution") -> go.Figure:
    """Rutas de una solution.VRPSolution con hover rico por nodo.

    Cada nodo del recorrido muestra: número de cliente, ruta, posición
    dentro de la ruta, demanda y carga acumulada hasta ese punto.
    """
    from app.models.solution import route_load

    instance = sol.instance
    fig = go.Figure()

    # --- Depósito ---
    fig.add_trace(go.Scatter(
        x=[instance.coords[0, 0]], y=[instance.coords[0, 1]],
        mode="markers+text", text=["Depósito"], textposition="top center",
        marker=dict(size=20, color="#2c3e50", symbol="square"),
        name="Depósito",
        hovertemplate=(
            "<b>Depósito</b><br>"
            "x=%{x:.2f}, y=%{y:.2f}<extra></extra>"
        ),
    ))

    # --- Clientes de fondo (todos, sin ruta) ---
    hover_bg = [
        f"<b>C{i}</b><br>Demanda: {instance.demands[i]:.1f}<br>"
        f"x={instance.coords[i,0]:.2f}, y={instance.coords[i,1]:.2f}"
        for i in range(1, instance.n_nodes)
    ]
    fig.add_trace(go.Scatter(
        x=instance.coords[1:, 0], y=instance.coords[1:, 1],
        mode="markers",
        marker=dict(size=9, color="#bdc3c7", line=dict(width=1, color="#7f8c8d")),
        name="Clientes",
        customdata=hover_bg,
        hovertemplate="%{customdata}<extra></extra>",
    ))

    # --- Rutas con hover por nodo ---
    utils = sol.capacity_utilizations()
    for idx, route in enumerate(sol.routes):
        color = COLORS[idx % len(COLORS)]
        load_total = route_load(route, instance)
        pct = utils[idx]
        nodes = [0] + route + [0]

        # Construir hover individualizado por nodo del recorrido
        cumload = 0.0
        customdata: list[str] = []
        for k, n in enumerate(nodes):
            if n == 0:
                if k == 0:
                    customdata.append(
                        f"<b>Depósito</b> — inicio Ruta {idx+1}<br>"
                        f"Carga en vehículo: 0 / {instance.capacity}"
                    )
                else:
                    customdata.append(
                        f"<b>Depósito</b> — fin Ruta {idx+1}<br>"
                        f"Carga total: {load_total:.1f} / {instance.capacity} ({pct:.1f}%)"
                    )
            else:
                cumload += instance.demands[n]
                pos = k  # posición (1-based en el recorrido real)
                customdata.append(
                    f"<b>C{n}</b> — Ruta {idx+1}, parada {pos}<br>"
                    f"Demanda: {instance.demands[n]:.1f}<br>"
                    f"Carga acumulada: {cumload:.1f} / {instance.capacity} "
                    f"({cumload / instance.capacity * 100:.1f}%)"
                )

        fig.add_trace(go.Scatter(
            x=[instance.coords[n, 0] for n in nodes],
            y=[instance.coords[n, 1] for n in nodes],
            mode="lines+markers",
            line=dict(color=color, width=2.4),
            marker=dict(size=8, color=color, line=dict(width=1, color="white")),
            name=f"Ruta {idx+1}  ({pct:.0f}%  |  {load_total:.1f}/{instance.capacity})",
            customdata=customdata,
            hovertemplate="%{customdata}<extra></extra>",
        ))

    feasible, _ = sol.feasibility_report()
    status = "✓ Factible" if feasible else "✗ Infactible"
    fig.update_layout(
        title=dict(
            text=f"{sol.method}  |  Costo: {sol.total_cost():.2f}  |  "
                 f"{sol.n_vehicles()} vehículos  |  {status}",
            font=dict(size=14),
        ),
        xaxis_title="Coordenada X",
        yaxis_title="Coordenada Y",
        legend_title="Rutas (utilización)",
        legend=dict(bgcolor="rgba(255,255,255,0.85)", borderwidth=1),
        height=540,
        hovermode="closest",
        plot_bgcolor="#f8f9fa",
        paper_bgcolor="white",
    )
    return fig


def plot_utilization_bars(sol: "NewVRPSolution") -> go.Figure:
    """Barras horizontales de utilización de capacidad Q por ruta.

    Verde ≤ 80 % · Naranja ≤ 95 % · Rojo > 95 %.
    """
    from app.models.solution import route_load

    utils = sol.capacity_utilizations()
    loads = [route_load(r, sol.instance) for r in sol.routes]
    labels = [f"Ruta {i + 1}" for i in range(sol.n_vehicles())]

    bar_colors = [
        "#27ae60" if u <= 80 else "#e67e22" if u <= 95 else "#e74c3c"
        for u in utils
    ]
    customdata = [
        f"Ruta {i+1}<br>Carga: {l:.1f} / {sol.instance.capacity}<br>Utilización: {u:.1f}%"
        for i, (u, l) in enumerate(zip(utils, loads))
    ]

    fig = go.Figure(go.Bar(
        y=labels, x=utils, orientation="h",
        marker_color=bar_colors,
        text=[f"{u:.1f}%  ({l:.1f}/{sol.instance.capacity})"
              for u, l in zip(utils, loads)],
        textposition="auto",
        customdata=customdata,
        hovertemplate="%{customdata}<extra></extra>",
    ))
    fig.add_vline(x=100, line_dash="dot", line_color="#e74c3c",
                  annotation_text="Q = 100 %", annotation_position="top right")
    fig.update_layout(
        title="Utilización de capacidad por ruta",
        xaxis=dict(title="Utilización (%)", range=[0, 118]),
        yaxis_title="",
        height=max(180, sol.n_vehicles() * 48 + 80),
        showlegend=False,
        plot_bgcolor="#f8f9fa",
        paper_bgcolor="white",
    )
    return fig


def compare_solutions_table(solutions: list["NewVRPSolution"]) -> pd.DataFrame:
    """DataFrame comparativo para una lista de solution.VRPSolution."""
    rows = []
    for sol in solutions:
        ok, _ = sol.feasibility_report()
        utils = sol.capacity_utilizations()
        rows.append({
            "Método": sol.method,
            "Costo total": sol.total_cost(),
            "Vehículos": sol.n_vehicles(),
            "Utilización prom. (%)": round(sum(utils) / len(utils), 1) if utils else 0.0,
            "Tiempo (s)": round(sol.metadata.get("computation_time_s", 0.0), 4),
            "Factible": "✓" if ok else "✗",
        })
    return pd.DataFrame(rows).sort_values("Costo total").reset_index(drop=True)


def plot_solutions_comparison(solutions: list["NewVRPSolution"]) -> go.Figure:
    """Barras de costo total por algoritmo, ordenadas ascendentemente."""
    df = compare_solutions_table(solutions)
    fig = px.bar(
        df, x="Método", y="Costo total",
        color="Método", text="Costo total",
        title="Comparación de costo total por algoritmo",
        color_discrete_sequence=COLORS,
        hover_data={"Vehículos": True, "Utilización prom. (%)": True, "Tiempo (s)": True},
    )
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig.update_layout(
        showlegend=False, height=400,
        plot_bgcolor="#f8f9fa", paper_bgcolor="white",
        yaxis_title="Costo total (distancia)",
    )
    return fig


def build_html_report(
    sol: "NewVRPSolution",
    fig: go.Figure,
    instance_params: dict | None = None,
) -> str:
    """Reporte HTML autocontenido con instancia, parámetros, métricas y gráfico.

    Parameters
    ----------
    sol : solution.VRPSolution
    fig : go.Figure  — gráfico de rutas ya generado
    instance_params : dict, optional — parámetros de generación de la instancia
    """
    import numpy as np

    inst = sol.instance
    feasible, errors = sol.feasibility_report()
    feasibility_badge = (
        '<span class="badge ok">✓ Factible</span>'
        if feasible
        else '<span class="badge err">✗ Infactible</span> '
        + " · ".join(errors)
    )

    # --- Tablas HTML ---
    def _df_to_html(df: pd.DataFrame) -> str:
        return df.to_html(index=False, border=0, classes="data-table")

    utils = sol.capacity_utilizations()
    summary_df = pd.DataFrame([{
        "Método": sol.method,
        "Costo total": sol.total_cost(),
        "Vehículos usados": sol.n_vehicles(),
        "Utilización prom. (%)": round(sum(utils) / len(utils), 1) if utils else 0.0,
        "Tiempo cómputo (s)": round(sol.metadata.get("computation_time_s", 0.0), 4),
    }])

    instance_df = pd.DataFrame([{
        "Nombre": inst.name,
        "Clientes (n)": inst.n_customers,
        "Capacidad (Q)": inst.capacity,
        "Depósito": f"({inst.coords[0,0]:.2f}, {inst.coords[0,1]:.2f})",
        "Demanda total": round(float(inst.demands[1:].sum()), 2),
        "Demanda mín.": round(float(inst.demands[1:].min()), 2),
        "Demanda máx.": round(float(inst.demands[1:].max()), 2),
        "Demanda prom.": round(float(inst.demands[1:].mean()), 2),
    }])

    # Parámetros del método
    params_rows = {k: v for k, v in sol.metadata.items()
                   if k not in ("computation_time_s",)}
    if instance_params:
        params_rows.update(instance_params)
    params_df = pd.DataFrame([{"Parámetro": k, "Valor": str(v)}
                               for k, v in params_rows.items()])

    routes_df = sol.to_routes_dataframe()
    chart_html = fig.to_html(full_html=False, include_plotlyjs="cdn")
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    return f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Reporte CVRP — {inst.name}</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body   {{ font-family: "Segoe UI", Arial, sans-serif; color: #2c3e50;
              background: #f4f6f8; padding: 32px 40px; }}
    header {{ background: #1a4e8a; color: white; padding: 20px 28px;
              border-radius: 8px; margin-bottom: 28px; }}
    header h1 {{ font-size: 1.6rem; margin-bottom: 4px; }}
    header p  {{ font-size: 0.85rem; opacity: .8; }}
    section   {{ background: white; border-radius: 8px; padding: 22px 28px;
                margin-bottom: 22px;
                box-shadow: 0 1px 4px rgba(0,0,0,.08); }}
    h2 {{ font-size: 1.1rem; color: #1a4e8a; border-bottom: 2px solid #e8eaf0;
          padding-bottom: 8px; margin-bottom: 16px; }}
    .data-table {{ border-collapse: collapse; width: 100%; font-size: .88rem; }}
    .data-table th {{ background: #eef2f7; color: #34495e; font-weight: 600;
                      padding: 8px 12px; text-align: left;
                      border-bottom: 2px solid #d5dbe8; }}
    .data-table td {{ padding: 7px 12px; border-bottom: 1px solid #eef0f5; }}
    .data-table tr:hover td {{ background: #f7f9fc; }}
    .badge    {{ display: inline-block; padding: 3px 10px; border-radius: 12px;
                 font-size: .85rem; font-weight: 700; }}
    .badge.ok {{ background: #d5f5e3; color: #1e8449; }}
    .badge.err {{ background: #fde8e8; color: #c0392b; }}
    .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(140px,1fr));
                    gap: 12px; margin-bottom: 4px; }}
    .metric {{ background: #eef2f7; border-radius: 6px; padding: 14px 16px; text-align: center; }}
    .metric .val {{ font-size: 1.5rem; font-weight: 700; color: #1a4e8a; }}
    .metric .lbl {{ font-size: .75rem; color: #7f8c8d; margin-top: 2px; }}
    @media print {{
      body {{ background: white; padding: 0; }}
      section {{ box-shadow: none; border: 1px solid #ddd; page-break-inside: avoid; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Reporte CVRP — {inst.name}</h1>
    <p>Generado: {now} &nbsp;·&nbsp; Método: {sol.method}</p>
  </header>

  <section>
    <h2>Instancia</h2>
    {_df_to_html(instance_df)}
  </section>

  <section>
    <h2>Parámetros del método</h2>
    {_df_to_html(params_df) if not params_df.empty else "<p>Sin parámetros configurables.</p>"}
  </section>

  <section>
    <h2>Métricas de la solución</h2>
    <div class="metric-grid">
      <div class="metric"><div class="val">{sol.total_cost():.2f}</div><div class="lbl">Costo total</div></div>
      <div class="metric"><div class="val">{sol.n_vehicles()}</div><div class="lbl">Vehículos</div></div>
      <div class="metric"><div class="val">{round(sum(utils)/len(utils),1) if utils else 0:.1f}%</div><div class="lbl">Utilización prom.</div></div>
      <div class="metric"><div class="val">{round(sol.metadata.get("computation_time_s",0),4):.4f}s</div><div class="lbl">Tiempo</div></div>
    </div>
    <p style="margin-top:12px">Factibilidad: {feasibility_badge}</p>
  </section>

  <section>
    <h2>Detalle de rutas</h2>
    {_df_to_html(routes_df)}
  </section>

  <section>
    <h2>Visualización</h2>
    {chart_html}
  </section>
</body>
</html>"""
