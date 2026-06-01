"""app.py — Herramienta CVRP (interfaz Streamlit de página única).

Ejecutar:
    streamlit run app.py

Este archivo solo contiene lógica de UI y orquestación.
Toda lógica algorítmica vive en el paquete `app`.
"""

from __future__ import annotations

import io
import os
import tempfile
from pathlib import Path

import streamlit as st

from app.algorithms.registry import METHOD_REGISTRY, PARAMS_SPEC
from app.models.instances import (
    VRPInstance,
    generate_cvrp_instance,
    plot_instance,
    read_instance_csv,
    write_instance_csv,
)
from app.utils.data_loader import csv_template
from app.utils.visualization import build_html_report, plot_cvrp_solution

# ---------------------------------------------------------------------------
# Configuración de página
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="CVRP Optimizer",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = Path(__file__).parent / "data" / "instances"

# ---------------------------------------------------------------------------
# Helpers de UI (sin lógica algorítmica)
# ---------------------------------------------------------------------------

def _render_param_widgets(method_name: str) -> dict:
    """Renderiza los widgets de parámetros según PARAMS_SPEC y devuelve los valores."""
    spec = PARAMS_SPEC.get(method_name, [])
    if not spec:
        st.caption("Este método no tiene parámetros configurables.")
        return {}
    values: dict = {}
    for p in spec:
        key = f"param_{method_name}_{p['key']}"
        if p["widget"] == "selectbox":
            default_idx = p["options"].index(p["default"]) if p["default"] in p["options"] else 0
            values[p["key"]] = st.selectbox(
                p["label"],
                options=p["options"],
                index=default_idx,
                help=p.get("help"),
                key=key,
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
                p["label"],
                value=p["default"],
                help=p.get("help"),
                key=key,
            )
    return values


def _find_instance_csvs() -> list[Path]:
    """Devuelve todos los CSV en data/instances (recursivo)."""
    if not DATA_DIR.exists():
        return []
    return sorted(DATA_DIR.rglob("*.csv"))


def _load_instance_from_path(path: Path) -> VRPInstance | None:
    """Carga una VRPInstance desde un archivo CSV; retorna None si falla."""
    try:
        return read_instance_csv(path)
    except (ValueError, FileNotFoundError) as e:
        st.error(f"Error al leer {path.name}: {e}")
        return None


def _load_instance_from_upload(uploaded_file) -> VRPInstance | None:
    """Carga una VRPInstance desde un st.file_uploader object."""
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="wb") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    try:
        return read_instance_csv(tmp_path)
    except (ValueError, FileNotFoundError) as e:
        st.error(f"Error de validación: {e}")
        return None
    finally:
        os.unlink(tmp_path)


def _save_instance_to_disk(instance: VRPInstance, subfolder: str = "small") -> Path:
    """Guarda la instancia en data/instances/<subfolder>/<name>.csv."""
    dest = DATA_DIR / subfolder / f"{instance.name}.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    write_instance_csv(instance, dest)
    return dest


def _csv_download_bytes(instance: VRPInstance) -> bytes:
    """Serializa la instancia como CSV en memoria."""
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="wb") as tmp:
        tmp_path = tmp.name
    write_instance_csv(instance, tmp_path)
    data = Path(tmp_path).read_bytes()
    os.unlink(tmp_path)
    return data


def _build_solution_csv(sol) -> bytes:
    df = sol.to_routes_dataframe()
    return df.to_csv(index=False).encode("utf-8")


# ---------------------------------------------------------------------------
# SIDEBAR — Instancia
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🚚 CVRP Optimizer")
    st.markdown("---")

    # ------------------------------------------------------------------ #
    # 1. Fuente de la instancia                                            #
    # ------------------------------------------------------------------ #
    st.subheader("📦 Instancia")
    source = st.radio(
        "Fuente",
        ["📁 Desde disco", "⬆️ Subir CSV", "🎲 Generar aleatoria"],
        label_visibility="collapsed",
    )

    instance: VRPInstance | None = st.session_state.get("instance")

    # --- Desde disco ---
    if source == "📁 Desde disco":
        csv_files = _find_instance_csvs()
        if not csv_files:
            st.info(
                "No hay archivos CSV en `data/instances/`.  \n"
                "Genera una instancia y guárdala en disco primero."
            )
        else:
            labels = {f.stem: f for f in csv_files}
            chosen_label = st.selectbox("Archivo:", list(labels.keys()))
            chosen_path = labels[chosen_label]
            if st.button("📂 Cargar archivo", use_container_width=True):
                inst = _load_instance_from_path(chosen_path)
                if inst:
                    st.session_state["instance"] = inst
                    st.session_state.pop("solution", None)
                    st.success(f"'{inst.name}' cargada.")
                    st.rerun()

    # --- Subir CSV ---
    elif source == "⬆️ Subir CSV":
        template_bytes = csv_template().to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Descargar plantilla CSV",
            data=template_bytes,
            file_name="plantilla_vrp.csv",
            mime="text/csv",
            use_container_width=True,
        )
        uploaded = st.file_uploader(
            "Sube tu CSV",
            type=["csv"],
            help="El CSV debe tener columnas: node, x, y, demand, capacity, name",
        )
        if uploaded and st.button("⬆️ Cargar CSV", use_container_width=True):
            inst = _load_instance_from_upload(uploaded)
            if inst:
                st.session_state["instance"] = inst
                st.session_state.pop("solution", None)
                st.success(f"'{inst.name}' cargada — {inst.n_customers} clientes, Q={inst.capacity}")
                st.rerun()

    # --- Generar aleatoria ---
    else:
        n_customers = st.slider("Clientes", 4, 100, 15)
        capacity = st.number_input("Capacidad Q", min_value=5.0, value=50.0, step=5.0)
        demand_max = st.slider("Demanda máx. por cliente", 1, 50, 15)
        coord_max = st.slider("Rango coordenadas (0–N)", 50, 500, 100)
        seed_gen = st.number_input("Semilla generación", min_value=0, value=42, step=1)
        name_input = st.text_input("Nombre de instancia", value="")

        col_gen, col_save = st.columns(2)
        with col_gen:
            if st.button("🎲 Generar", use_container_width=True):
                try:
                    inst = generate_cvrp_instance(
                        n_customers=n_customers,
                        capacity=capacity,
                        coord_range=(0.0, float(coord_max)),
                        demand_range=(1.0, float(demand_max)),
                        seed=int(seed_gen),
                        name=name_input.strip() or None,
                    )
                    st.session_state["instance"] = inst
                    st.session_state.pop("solution", None)
                    st.rerun()
                except ValueError as e:
                    st.error(str(e))

        instance_now = st.session_state.get("instance")
        with col_save:
            if instance_now and st.button("💾 Guardar", use_container_width=True):
                subfolder = (
                    "small" if instance_now.n_customers <= 15
                    else "medium" if instance_now.n_customers <= 40
                    else "large"
                )
                dest = _save_instance_to_disk(instance_now, subfolder)
                st.success(f"Guardado en `{dest.relative_to(Path.cwd())}`")

    # ------------------------------------------------------------------ #
    # 2. Descarga de instancia generada / cargada                         #
    # ------------------------------------------------------------------ #
    instance = st.session_state.get("instance")
    if instance and source in ("🎲 Generar aleatoria",):
        st.download_button(
            "⬇️ Descargar instancia CSV",
            data=_csv_download_bytes(instance),
            file_name=f"{instance.name}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.markdown("---")

    # ------------------------------------------------------------------ #
    # 3. Método y parámetros                                               #
    # ------------------------------------------------------------------ #
    st.subheader("⚙️ Método")
    method_name = st.selectbox("Algoritmo:", list(METHOD_REGISTRY.keys()))
    param_values = _render_param_widgets(method_name)

    st.markdown("---")

    # ------------------------------------------------------------------ #
    # 4. Botón Resolver                                                    #
    # ------------------------------------------------------------------ #
    instance = st.session_state.get("instance")
    solve_disabled = instance is None
    if st.button(
        "▶️ Resolver",
        use_container_width=True,
        type="primary",
        disabled=solve_disabled,
    ):
        with st.spinner("Resolviendo…"):
            solver_cls = METHOD_REGISTRY[method_name]
            # Extraer seed del param_values si existe (para pasarlo aparte)
            kw = {k: v for k, v in param_values.items() if k != "seed"}
            seed_val = param_values.get("seed", None)
            solver = solver_cls(**kw)
            sol = solver.solve(instance, seed=seed_val)
        st.session_state["solution"] = sol
        st.rerun()

    if solve_disabled:
        st.caption("Carga una instancia primero.")

# ---------------------------------------------------------------------------
# ÁREA PRINCIPAL
# ---------------------------------------------------------------------------

instance = st.session_state.get("instance")
solution = st.session_state.get("solution")

# --- Sin instancia: bienvenida ---
if instance is None:
    st.markdown(
        """
        ## 🚚 CVRP Optimizer

        Herramienta para resolver el **Capacitated Vehicle Routing Problem**.

        **¿Cómo empezar?**

        | Paso | Acción |
        |------|--------|
        | 1 | Elige una fuente de instancia en el panel izquierdo |
        | 2 | Selecciona el algoritmo y configura sus parámetros |
        | 3 | Haz clic en **▶ Resolver** |
        | 4 | Descarga los resultados en CSV o HTML |

        ---
        **Convención**: una ruta `[1, 4, 7]` representa el recorrido `0 → 1 → 4 → 7 → 0`,
        donde el nodo **0** es siempre el depósito.
        """
    )
    st.stop()

# --- Con instancia: header de estado ---
st.markdown(
    f"**Instancia activa:** {instance.name} &nbsp;|&nbsp; "
    f"{instance.n_customers} clientes &nbsp;|&nbsp; Q = {instance.capacity}"
)
st.markdown("---")

# --- Tabs: Instancia / Resultados ---
tab_inst, tab_res = st.tabs(["🗺️ Instancia", "📊 Resultados"])

# ------------------------------------------------------------------ #
# Tab 1: Instancia                                                     #
# ------------------------------------------------------------------ #
with tab_inst:
    col_info, col_plot = st.columns([1, 2])
    with col_info:
        st.subheader("Resumen")
        summary = instance.summary()
        for k, v in summary.items():
            st.metric(label=k, value=str(v))
    with col_plot:
        st.subheader("Mapa de nodos")
        st.plotly_chart(plot_instance(instance), use_container_width=True)

# ------------------------------------------------------------------ #
# Tab 2: Resultados                                                    #
# ------------------------------------------------------------------ #
with tab_res:
    if solution is None:
        st.info("Configura el método en el panel izquierdo y haz clic en **▶ Resolver**.")
        st.stop()

    # --- Verificar que la solución corresponde a la instancia activa ---
    if solution.instance is not instance:
        st.warning(
            "La solución mostrada corresponde a una instancia anterior. "
            "Haz clic en **▶ Resolver** para actualizar."
        )

    # --- Métricas principales ---
    feasible, feas_errors = solution.feasibility_report()
    elapsed = solution.metadata.get("computation_time_s", 0.0)
    utils = solution.capacity_utilizations()
    avg_util = round(sum(utils) / len(utils), 1) if utils else 0.0

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Costo total", f"{solution.total_cost():.2f}")
    col2.metric("Vehículos", solution.n_vehicles())
    col3.metric("Utilización prom.", f"{avg_util:.1f}%")
    col4.metric("Tiempo (s)", f"{elapsed:.4f}")
    col5.metric(
        "Factibilidad",
        "✓ Factible" if feasible else "✗ Infactible",
        delta=None,
    )

    # --- Reporte de factibilidad (si hay errores) ---
    if not feasible:
        with st.expander("⚠️ Errores de factibilidad", expanded=True):
            for err in feas_errors:
                st.error(err)

    # --- Gráfico de rutas ---
    st.subheader("Rutas")
    fig = plot_cvrp_solution(solution)
    st.plotly_chart(fig, use_container_width=True)

    # --- Tabla de rutas ---
    st.subheader("Detalle por ruta")
    df_routes = solution.to_routes_dataframe()
    st.dataframe(df_routes, use_container_width=True, hide_index=True)

    # --- Detalle textual expandible ---
    with st.expander("Ver secuencias completas"):
        for _, row in df_routes.iterrows():
            st.markdown(
                f"**Ruta {int(row['Ruta'])}:** {row['Secuencia']}  \n"
                f"Carga: `{row['Carga']}/{instance.capacity}` "
                f"({row['Utilización (%)']:.1f}%)  |  "
                f"Costo: `{row['Costo']:.4f}`"
            )

    # --- Historial de construcción (solo Nearest Neighbor) ---
    if solution.history:
        with st.expander(f"📋 Historial de construcción ({len(solution.history)} eventos)"):
            import pandas as pd
            steps = [h for h in solution.history if h["event"] == "step"]
            if steps:
                st.dataframe(
                    pd.DataFrame(steps).drop(columns=["event"]),
                    use_container_width=True,
                    hide_index=True,
                )

    # --- Descargas ---
    st.subheader("Descargar resultados")
    dl_col1, dl_col2 = st.columns(2)

    with dl_col1:
        csv_bytes = _build_solution_csv(solution)
        st.download_button(
            "⬇️ Resultados CSV",
            data=csv_bytes,
            file_name=f"solucion_{solution.method.replace(' ', '_')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with dl_col2:
        html_report = build_html_report(solution, fig)
        st.download_button(
            "⬇️ Reporte HTML",
            data=html_report.encode("utf-8"),
            file_name=f"reporte_{solution.method.replace(' ', '_')}.html",
            mime="text/html",
            use_container_width=True,
        )
