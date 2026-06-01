"""app.py — CVRP Optimizer · Interfaz Streamlit de página única.

Ejecución:
    streamlit run app.py

Separación de responsabilidades:
  · Este archivo contiene SOLO lógica de UI y orquestación.
  · Toda lógica matemática vive en el paquete `app/`.
  · @st.cache_data se aplica a funciones de visualización puras (costosas
    en reruns) sin tocar los modelos ni los solvers.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st

from app.algorithms.registry import METHOD_DESCRIPTIONS, METHOD_REGISTRY, PARAMS_SPEC
from app.models.instances import (
    VRPInstance,
    generate_cvrp_instance,
    plot_instance,
    read_instance_csv,
    write_instance_csv,
)
from app.models.solution import VRPSolution
from app.utils.data_loader import csv_template
from app.utils.visualization import (
    build_html_report,
    compare_solutions_table,
    plot_cvrp_solution,
    plot_solutions_comparison,
    plot_utilization_bars,
)

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="CVRP Optimizer",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = Path(__file__).parent / "data" / "instances"

# ---------------------------------------------------------------------------
# Caching — funciones de visualización puras
# Los solvers y modelos matemáticos NO se modifican.
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _cached_plot_instance(
    coords_bytes: bytes, demands_bytes: bytes, capacity: float, name: str
):
    """Genera el mapa Plotly de la instancia; se cachea por contenido de coords."""
    coords = np.frombuffer(coords_bytes, dtype=np.float64).reshape(-1, 2).copy()
    demands = np.frombuffer(demands_bytes, dtype=np.float64).copy()
    inst = VRPInstance(coords=coords, demands=demands, capacity=capacity, name=name)
    return plot_instance(inst)


@st.cache_data(show_spinner=False)
def _cached_plot_solution(
    coords_bytes: bytes, demands_bytes: bytes, capacity: float, name: str,
    routes_json: str, method: str,
):
    """Genera el gráfico de rutas; se cachea mientras la solución no cambie."""
    coords = np.frombuffer(coords_bytes, dtype=np.float64).reshape(-1, 2).copy()
    demands = np.frombuffer(demands_bytes, dtype=np.float64).copy()
    inst = VRPInstance(coords=coords, demands=demands, capacity=capacity, name=name)
    sol = VRPSolution(instance=inst, routes=json.loads(routes_json), method=method)
    return plot_cvrp_solution(sol)


# ---------------------------------------------------------------------------
# Helpers de UI — sin lógica algorítmica
# ---------------------------------------------------------------------------

def _render_param_widgets(method_name: str) -> dict:
    """Renderiza los widgets de parámetros desde PARAMS_SPEC."""
    spec = PARAMS_SPEC.get(method_name, [])
    if not spec:
        st.caption("Sin parámetros configurables para este método.")
        return {}
    values: dict = {}
    for p in spec:
        key = f"param__{method_name}__{p['key']}"
        if p["widget"] == "selectbox":
            idx = p["options"].index(p["default"]) if p["default"] in p["options"] else 0
            values[p["key"]] = st.selectbox(
                p["label"], p["options"], index=idx, help=p.get("help"), key=key,
            )
        elif p["widget"] == "number_input":
            values[p["key"]] = st.number_input(
                p["label"],
                min_value=p.get("min_value", 0),
                max_value=p.get("max_value", 99999),
                value=p["default"],
                step=p.get("step", 1),
                help=p.get("help"),
                key=key,
            )
        elif p["widget"] == "checkbox":
            values[p["key"]] = st.checkbox(
                p["label"], value=p["default"], help=p.get("help"), key=key,
            )
    return values


def _find_csvs() -> list[Path]:
    if not DATA_DIR.exists():
        return []
    return sorted(DATA_DIR.rglob("*.csv"))


def _load_from_path(path: Path) -> VRPInstance | None:
    try:
        return read_instance_csv(path)
    except FileNotFoundError:
        st.error(
            f"**Archivo no encontrado:** `{path.name}`  \n"
            "El archivo fue eliminado o movido. Recarga la página."
        )
    except ValueError as e:
        st.error(
            f"**Error al leer `{path.name}`:** {e}  \n"
            "Verifica que el CSV tiene las columnas `node, x, y, demand, capacity, name`."
        )
    return None


def _load_from_upload(uploaded) -> VRPInstance | None:
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="wb") as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name
    try:
        return read_instance_csv(tmp_path)
    except ValueError as e:
        msg = str(e)
        if "Faltan" in msg:
            st.error(
                f"**Columnas obligatorias faltantes:** {msg}  \n"
                "Descarga la plantilla CSV para ver el formato correcto."
            )
        elif "negativa" in msg:
            st.error("**Demandas negativas detectadas.** Todas las demandas deben ser ≥ 0.")
        elif "capacidad" in msg:
            st.error(
                "**Demanda mayor que la capacidad Q.** "
                "Revisa que ningún cliente tenga demanda > Q."
            )
        elif "depósito" in msg.lower():
            st.error(
                "**El nodo 0 (depósito) debe tener demanda = 0.** "
                "Ajusta la columna `demand` del nodo 0."
            )
        else:
            st.error(f"**Error de validación:** {e}")
    except Exception as e:
        st.error(f"**Error inesperado al leer el archivo:** {e}")
    finally:
        os.unlink(tmp_path)
    return None


def _save_to_disk(inst: VRPInstance) -> Path:
    sub = "small" if inst.n_customers <= 15 else "medium" if inst.n_customers <= 40 else "large"
    dest = DATA_DIR / sub / f"{inst.name}.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    write_instance_csv(inst, dest)
    return dest


def _instance_csv_bytes(inst: VRPInstance) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        tmp_path = tmp.name
    write_instance_csv(inst, tmp_path)
    data = Path(tmp_path).read_bytes()
    os.unlink(tmp_path)
    return data


def _run_solve(inst: VRPInstance, method_name: str, param_values: dict) -> VRPSolution:
    """Ejecuta el solver y devuelve un VRPSolution. Sin caché (rápido para heurísticas)."""
    kw = {k: v for k, v in param_values.items() if k != "seed"}
    seed = param_values.get("seed", None)
    solver = METHOD_REGISTRY[method_name](**kw)
    return solver.solve(inst, seed=seed)


# ---------------------------------------------------------------------------
# SIDEBAR — estado activo
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 🚚 CVRP Optimizer")
    st.caption("Herramienta educativa de ruteo con capacidad.")
    st.markdown("---")

    inst_sidebar = st.session_state.get("instance")
    if inst_sidebar is None:
        st.info("Sin instancia cargada.  \nVe a la pestaña **Instancia** para comenzar.")
    else:
        st.success(
            f"**{inst_sidebar.name}**  \n"
            f"{inst_sidebar.n_customers} clientes · Q = {inst_sidebar.capacity}"
        )
        sol_sidebar = st.session_state.get("solution")
        if sol_sidebar:
            ok, _ = sol_sidebar.feasibility_report()
            st.metric("Último costo", f"{sol_sidebar.total_cost():.2f}")
            st.metric("Vehículos", sol_sidebar.n_vehicles())
            st.caption(f"Método: {sol_sidebar.method} · {'✓ Factible' if ok else '✗ Infactible'}")

    st.markdown("---")
    st.caption(
        "**Convención de rutas:**  \n"
        "`[1, 4, 7]` = 0 → 1 → 4 → 7 → 0  \n"
        "El nodo **0** es siempre el depósito."
    )

# ---------------------------------------------------------------------------
# ÁREA PRINCIPAL — bienvenida sin instancia
# ---------------------------------------------------------------------------

instance: VRPInstance | None = st.session_state.get("instance")
solution: VRPSolution | None = st.session_state.get("solution")

if instance is None:
    st.markdown("""
# 🚚 CVRP Optimizer

Herramienta para resolver el **Capacitated Vehicle Routing Problem (CVRP)**
usando heurísticas clásicas de investigación de operaciones.

---

### ¿Cómo empezar?

| Paso | Pestaña | Acción |
|------|---------|--------|
| 1 | **Instancia** | Carga, sube o genera una instancia |
| 2 | **Resolver** | Elige un algoritmo y ejecútalo |
| 3 | **Comparar** | Compara varios algoritmos en la misma instancia |
| 4 | **Exportar** | Descarga los resultados en CSV o HTML |

---

> **Convención:** una ruta `[1, 4, 7]` representa el recorrido
> `0 → 1 → 4 → 7 → 0`, donde el nodo **0** es siempre el depósito.
""")
    st.info("👈 Comienza cargando una instancia en la pestaña **Instancia**.")

# ---------------------------------------------------------------------------
# TABS PRINCIPALES
# ---------------------------------------------------------------------------

tab_inst, tab_res, tab_cmp, tab_exp = st.tabs(
    ["🗺️ Instancia", "⚙️ Resolver", "📊 Comparar", "⬇️ Exportar"]
)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — INSTANCIA
# ═══════════════════════════════════════════════════════════════════════════
with tab_inst:
    col_ctrl, col_map = st.columns([1, 2], gap="large")

    with col_ctrl:
        st.subheader("Fuente de datos")
        source = st.radio(
            "Origen",
            ["📁 Desde disco", "⬆️ Subir CSV", "🎲 Generar aleatoria"],
            label_visibility="collapsed",
        )

        # ------ Desde disco ------
        if source == "📁 Desde disco":
            csv_files = _find_csvs()
            if not csv_files:
                st.info(
                    "No hay archivos CSV en `data/instances/`.  \n"
                    "Genera una instancia y pulsa **💾 Guardar en disco**."
                )
            else:
                labels = {f"{f.parent.name}/{f.stem}": f for f in csv_files}
                chosen = st.selectbox(
                    "Archivo:", list(labels.keys()),
                    help="Selecciona una instancia guardada previamente.",
                )
                if st.button("📂 Cargar", use_container_width=True):
                    inst = _load_from_path(labels[chosen])
                    if inst:
                        st.session_state["instance"] = inst
                        st.session_state.pop("solution", None)
                        st.success(f"'{inst.name}' cargada — {inst.n_customers} clientes.")
                        st.rerun()

        # ------ Subir CSV ------
        elif source == "⬆️ Subir CSV":
            with st.expander("📋 Formato esperado del CSV", expanded=False):
                st.markdown(
                    "El archivo debe tener exactamente estas columnas:\n\n"
                    "| Columna | Tipo | Descripción |\n"
                    "|---------|------|-------------|\n"
                    "| `node` | int | Índice de nodo; **0 = depósito** |\n"
                    "| `x` | float | Coordenada X |\n"
                    "| `y` | float | Coordenada Y |\n"
                    "| `demand` | float | Demanda (0 para el depósito) |\n"
                    "| `capacity` | float | Capacidad Q (mismo valor en todas las filas) |\n"
                    "| `name` | str | Nombre de la instancia |"
                )
            st.download_button(
                "⬇️ Descargar plantilla CSV",
                data=csv_template().to_csv(index=False).encode("utf-8"),
                file_name="plantilla_vrp.csv", mime="text/csv",
                use_container_width=True,
            )
            uploaded = st.file_uploader(
                "Sube tu CSV", type=["csv"],
                help="Solo archivos .csv con el formato indicado arriba.",
            )
            if uploaded:
                if st.button("⬆️ Cargar CSV", use_container_width=True):
                    inst = _load_from_upload(uploaded)
                    if inst:
                        st.session_state["instance"] = inst
                        st.session_state.pop("solution", None)
                        st.success(
                            f"'{inst.name}' cargada — "
                            f"{inst.n_customers} clientes, Q = {inst.capacity}"
                        )
                        st.rerun()

        # ------ Generar aleatoria ------
        else:
            with st.form("form_generate"):
                n_cust = st.slider(
                    "Número de clientes", 4, 100, 15,
                    help="Cuántos nodos cliente tendrá la instancia (excluye el depósito).",
                )
                cap = st.number_input(
                    "Capacidad por vehículo (Q)", min_value=5.0, value=50.0, step=5.0,
                    help="Demanda máxima que puede cargar cada vehículo.",
                )
                d_max = st.slider(
                    "Demanda máx. por cliente", 1, 50, 15,
                    help="Límite superior de la demanda individual; se recorta a Q si supera.",
                )
                c_max = st.slider("Rango de coordenadas (0 – N)", 50, 500, 100)
                seed_val = st.number_input(
                    "Semilla", min_value=0, value=42, step=1,
                    help="Misma semilla → misma instancia. Útil para experimentos reproducibles.",
                )
                name_val = st.text_input("Nombre (opcional)", value="")
                submitted = st.form_submit_button("🎲 Generar", use_container_width=True)

            if submitted:
                try:
                    inst = generate_cvrp_instance(
                        n_customers=n_cust, capacity=cap,
                        coord_range=(0.0, float(c_max)),
                        demand_range=(1.0, float(d_max)),
                        seed=int(seed_val),
                        name=name_val.strip() or None,
                    )
                    st.session_state["instance"] = inst
                    st.session_state.pop("solution", None)
                    st.success(f"Instancia '{inst.name}' generada.")
                    st.rerun()
                except ValueError as e:
                    st.error(f"**Parámetros inválidos:** {e}")

        # ------ Acciones sobre instancia activa ------
        instance = st.session_state.get("instance")
        if instance:
            st.markdown("---")
            st.caption(f"**Instancia activa:** {instance.name}")

            col_a, col_b = st.columns(2)
            with col_a:
                st.download_button(
                    "⬇️ CSV",
                    data=_instance_csv_bytes(instance),
                    file_name=f"{instance.name}.csv",
                    mime="text/csv", use_container_width=True,
                    help="Descarga esta instancia en formato CSV reutilizable.",
                )
            with col_b:
                if st.button("💾 Guardar en disco", use_container_width=True,
                             help="Guarda en data/instances/ para cargarla luego."):
                    dest = _save_to_disk(instance)
                    st.success(f"Guardado en `{dest.relative_to(Path.cwd())}`")

    with col_map:
        instance = st.session_state.get("instance")
        if instance is None:
            st.info("Carga o genera una instancia para ver el mapa.")
        else:
            st.subheader(f"Mapa — {instance.name}")
            fig_inst = _cached_plot_instance(
                instance.coords.tobytes(),
                instance.demands.tobytes(),
                instance.capacity,
                instance.name,
            )
            st.plotly_chart(fig_inst, use_container_width=True)

            # Estadísticas de la instancia en columnas compactas
            summ = instance.summary()
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Clientes (n)", summ["Clientes"])
            c2.metric("Capacidad (Q)", summ["Capacidad (Q)"])
            c3.metric("Demanda total", f"{summ['Demanda total']:.1f}")
            c4.metric("Vehículos mín. ↓", summ["Vehículos mínimos requeridos (cota inferior)"],
                      help="Cota inferior teórica: ⌈Demanda total / Q⌉")

# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — RESOLVER
# ═══════════════════════════════════════════════════════════════════════════
with tab_res:
    instance = st.session_state.get("instance")
    if instance is None:
        st.warning("⬅️ Carga una instancia en la pestaña **Instancia** para poder resolver.")
        st.stop()

    col_meth, col_out = st.columns([1, 2], gap="large")

    with col_meth:
        st.subheader("Configuración")
        method_name = st.selectbox(
            "Algoritmo",
            list(METHOD_REGISTRY.keys()),
            help="Elige el método con el que resolver la instancia activa.",
        )

        with st.expander("ℹ️ ¿Cómo funciona este algoritmo?", expanded=False):
            st.markdown(METHOD_DESCRIPTIONS.get(method_name, "_Sin descripción._"))

        st.markdown("**Parámetros**")
        param_values = _render_param_widgets(method_name)

        st.markdown("---")
        if st.button("▶️ Resolver", type="primary", use_container_width=True):
            with st.spinner(f"Ejecutando {method_name}…"):
                try:
                    sol = _run_solve(instance, method_name, param_values)
                    st.session_state["solution"] = sol
                    st.session_state.pop("comparison_solutions", None)
                    st.rerun()
                except Exception as e:
                    st.error(f"**Error al resolver:** {e}")

        st.caption(
            f"Instancia: **{instance.name}** · "
            f"{instance.n_customers} clientes · Q = {instance.capacity}"
        )

    with col_out:
        solution = st.session_state.get("solution")
        if solution is None:
            st.info("Configura el método y pulsa **▶️ Resolver** para ver los resultados.")
        else:
            if solution.instance is not instance:
                st.warning(
                    "⚠️ La solución fue generada con una instancia anterior.  \n"
                    "Pulsa **▶️ Resolver** para actualizar."
                )

            st.subheader("Resultados")
            feasible, feas_errors = solution.feasibility_report()
            elapsed = solution.metadata.get("computation_time_s", 0.0)
            utils = solution.capacity_utilizations()
            avg_util = round(sum(utils) / len(utils), 1) if utils else 0.0

            # Métricas principales
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Costo total", f"{solution.total_cost():.2f}",
                      help="Suma de distancias euclidianas de todas las rutas.")
            m2.metric("Vehículos", solution.n_vehicles(),
                      help="Número de rutas no vacías en la solución.")
            m3.metric("Utilización prom.", f"{avg_util:.1f}%",
                      help="Promedio de (carga / Q) × 100 por ruta.")
            m4.metric("Tiempo (s)", f"{elapsed:.4f}",
                      help="Tiempo de cómputo del solver.")
            m5.metric(
                "Factibilidad",
                "✓ Factible" if feasible else "✗ Infactible",
                help="Verifica: todos los clientes visitados una vez y carga ≤ Q.",
            )

            if not feasible:
                with st.expander("⚠️ Errores de factibilidad", expanded=True):
                    for err in feas_errors:
                        st.error(err)

            # Gráfico de rutas (cacheado)
            fig_sol = _cached_plot_solution(
                instance.coords.tobytes(),
                instance.demands.tobytes(),
                instance.capacity,
                instance.name,
                json.dumps(solution.routes),
                solution.method,
            )
            st.plotly_chart(fig_sol, use_container_width=True)

            # Barras de utilización
            with st.expander("📊 Utilización de capacidad por ruta", expanded=False):
                st.plotly_chart(
                    plot_utilization_bars(solution), use_container_width=True
                )

            # Tabla de rutas
            st.subheader("Detalle por ruta")
            df_routes = solution.to_routes_dataframe()
            st.dataframe(df_routes, use_container_width=True, hide_index=True)

            # Secuencias textuales
            with st.expander("Secuencias completas"):
                for _, row in df_routes.iterrows():
                    st.markdown(
                        f"**Ruta {int(row['Ruta'])}:** `{row['Secuencia']}`  \n"
                        f"Carga: **{row['Carga']}/{instance.capacity}** "
                        f"({row['Utilización (%)']:.1f}%) · "
                        f"Costo: **{row['Costo']:.4f}**"
                    )

            # Historial de construcción (Nearest Neighbor)
            if solution.history:
                steps = [h for h in solution.history if h["event"] == "step"]
                with st.expander(
                    f"📋 Historial de construcción — {len(steps)} pasos", expanded=False
                ):
                    import pandas as pd
                    st.info(
                        "Cada fila muestra un paso de construcción: "
                        "desde qué nodo se partió, adónde se fue, "
                        "la distancia recorrida y la carga acumulada."
                    )
                    st.dataframe(
                        pd.DataFrame(steps).drop(columns=["event"]),
                        use_container_width=True, hide_index=True,
                    )

# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — COMPARAR
# ═══════════════════════════════════════════════════════════════════════════
with tab_cmp:
    instance = st.session_state.get("instance")
    if instance is None:
        st.warning("⬅️ Carga una instancia en la pestaña **Instancia** para comparar.")
        st.stop()

    st.subheader("Comparar algoritmos")
    st.caption(f"Instancia: **{instance.name}** · {instance.n_customers} clientes · Q = {instance.capacity}")

    selected = st.multiselect(
        "Algoritmos a comparar",
        list(METHOD_REGISTRY.keys()),
        default=list(METHOD_REGISTRY.keys()),
        help="Selecciona uno o más algoritmos para ejecutar sobre la instancia activa.",
    )

    cmp_seed = st.number_input(
        "Semilla (para métodos aleatorios)", min_value=0, value=42, step=1,
        help="Semilla compartida para reproducibilidad en la comparación.",
    )

    if st.button("▶️ Comparar todos", type="primary", disabled=not selected):
        results: list[VRPSolution] = []
        prog = st.progress(0, text="Iniciando…")
        for i, name in enumerate(selected):
            prog.progress((i + 1) / len(selected), text=f"Ejecutando {name}…")
            try:
                sol = METHOD_REGISTRY[name]().solve(instance, seed=int(cmp_seed))
                results.append(sol)
            except Exception as e:
                st.error(f"**{name}** falló: {e}")
        prog.empty()
        if results:
            st.session_state["comparison_solutions"] = results
            st.rerun()

    cmp_solutions: list[VRPSolution] = st.session_state.get("comparison_solutions", [])
    if not cmp_solutions:
        st.info("Selecciona los algoritmos y pulsa **▶️ Comparar todos**.")
    else:
        st.subheader("Tabla comparativa")
        df_cmp = compare_solutions_table(cmp_solutions)
        st.dataframe(df_cmp, use_container_width=True, hide_index=True)

        best = df_cmp.iloc[0]["Método"]
        st.success(f"🏆 Mejor solución encontrada: **{best}** con costo **{df_cmp.iloc[0]['Costo total']:.2f}**")

        st.subheader("Gráfico comparativo")
        st.plotly_chart(plot_solutions_comparison(cmp_solutions), use_container_width=True)

        st.subheader("Rutas por algoritmo")
        cols = st.columns(min(len(cmp_solutions), 2))
        for i, sol in enumerate(cmp_solutions):
            with cols[i % 2]:
                fig_i = _cached_plot_solution(
                    instance.coords.tobytes(),
                    instance.demands.tobytes(),
                    instance.capacity,
                    instance.name,
                    json.dumps(sol.routes),
                    sol.method,
                )
                st.plotly_chart(fig_i, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 4 — EXPORTAR
# ═══════════════════════════════════════════════════════════════════════════
with tab_exp:
    instance = st.session_state.get("instance")
    solution = st.session_state.get("solution")

    if solution is None:
        st.warning("⬅️ Resuelve la instancia en la pestaña **Resolver** para exportar.")
        st.stop()

    st.subheader("Exportar resultados")
    feasible, _ = solution.feasibility_report()

    # Vista previa del contenido del reporte
    with st.expander("👁️ Vista previa del reporte HTML", expanded=False):
        st.markdown(f"""
El reporte incluirá:

| Sección | Contenido |
|---------|-----------|
| **Instancia** | Nombre, clientes, Q, coordenadas del depósito, estadísticas de demanda |
| **Parámetros** | Método, estrategia de inicio, desempate, semilla |
| **Métricas** | Costo total, vehículos, utilización, tiempo, factibilidad |
| **Rutas** | Tabla con secuencia, carga y costo por ruta |
| **Gráfico** | Mapa interactivo Plotly embebido (funciona sin internet si se incluye JS) |

> **Instancia activa:** {instance.name if instance else "—"}
> **Método:** {solution.method}
> **Costo:** {solution.total_cost():.2f} · **Factible:** {"✓" if feasible else "✗"}
""")

    # Construir los artefactos
    fig_exp = _cached_plot_solution(
        instance.coords.tobytes(),
        instance.demands.tobytes(),
        instance.capacity,
        instance.name,
        json.dumps(solution.routes),
        solution.method,
    )

    # Parámetros extra para el reporte
    instance_params = {
        "Nombre instancia": instance.name,
        "n_customers": instance.n_customers,
        "capacity (Q)": instance.capacity,
    }

    html_report = build_html_report(solution, fig_exp, instance_params)
    csv_bytes = solution.to_routes_dataframe().to_csv(index=False).encode("utf-8")

    safe_method = solution.method.replace(" ", "_").replace("&", "and")
    safe_name = instance.name.replace(" ", "_")

    col_d1, col_d2, col_d3 = st.columns(3)

    with col_d1:
        st.download_button(
            "⬇️ Rutas CSV",
            data=csv_bytes,
            file_name=f"rutas_{safe_name}_{safe_method}.csv",
            mime="text/csv",
            use_container_width=True,
            help="Tabla de rutas con carga, utilización y costo por ruta.",
        )

    with col_d2:
        st.download_button(
            "⬇️ Reporte HTML",
            data=html_report.encode("utf-8"),
            file_name=f"reporte_{safe_name}_{safe_method}.html",
            mime="text/html",
            use_container_width=True,
            help="Reporte autocontenido: se abre en cualquier navegador sin conexión.",
        )

    with col_d3:
        st.download_button(
            "⬇️ Instancia CSV",
            data=_instance_csv_bytes(instance),
            file_name=f"instancia_{safe_name}.csv",
            mime="text/csv",
            use_container_width=True,
            help="Instancia en formato reutilizable para cargarla en otra sesión.",
        )

    st.markdown("---")

    # Resumen final
    st.subheader("Resumen de la sesión")
    utils = solution.capacity_utilizations()
    st.dataframe(
        solution.to_routes_dataframe(),
        use_container_width=True,
        hide_index=True,
    )

    cmp_solutions = st.session_state.get("comparison_solutions", [])
    if cmp_solutions:
        st.subheader("Comparación de algoritmos ejecutados")
        st.dataframe(
            compare_solutions_table(cmp_solutions),
            use_container_width=True, hide_index=True,
        )
