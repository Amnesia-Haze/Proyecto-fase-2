import io

import pandas as pd
import streamlit as st

from app.models.instances import (
    generate_cvrp_instance,
    plot_instance,
    read_instance_csv,
    write_instance_csv,
)
from app.utils.data_loader import csv_template
from app.utils.preloaded_instances import PRELOADED

st.set_page_config(page_title="Instancias CVRP", layout="wide")
st.title("📦 Instancias CVRP")

tab1, tab2, tab3 = st.tabs(["Instancias precargadas", "Generar aleatoria", "Subir CSV"])

# ---------------------------------------------------------------------------
# Tab 1: Precargadas
# ---------------------------------------------------------------------------
with tab1:
    st.subheader("Selecciona una instancia de ejemplo")
    selected = st.selectbox("Instancia:", list(PRELOADED.keys()))
    instance = PRELOADED[selected]

    col_info, col_plot = st.columns([1, 2])
    with col_info:
        st.json(instance.summary())
    with col_plot:
        st.plotly_chart(plot_instance(instance), use_container_width=True)

    if st.button("Usar esta instancia", key="use_preloaded"):
        st.session_state["instance"] = instance
        st.success(f"Instancia '{instance.name}' cargada. Ve a **Resolver** para continuar.")

# ---------------------------------------------------------------------------
# Tab 2: Generar aleatoria
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("Generar instancia aleatoria")
    col1, col2, col3 = st.columns(3)
    with col1:
        n_customers = st.slider("Número de clientes", 5, 100, 15)
        seed = st.number_input("Semilla aleatoria", min_value=0, value=42, step=1)
    with col2:
        capacity = st.number_input("Capacidad por vehículo (Q)", min_value=10.0, value=50.0, step=5.0)
        demand_max = st.slider("Demanda máxima por cliente", 1, 50, 20)
    with col3:
        coord_max = st.slider("Rango coordenadas (0 a N)", 50, 500, 100)
        instance_name = st.text_input("Nombre (opcional)", value="")

    if st.button("Generar instancia"):
        try:
            instance = generate_cvrp_instance(
                n_customers=n_customers,
                capacity=capacity,
                coord_range=(0.0, float(coord_max)),
                demand_range=(1.0, float(demand_max)),
                seed=int(seed),
                name=instance_name or None,
            )
            st.session_state["instance"] = instance
            st.success("Instancia generada exitosamente.")

            col_info2, col_plot2 = st.columns([1, 2])
            with col_info2:
                st.json(instance.summary())
            with col_plot2:
                st.plotly_chart(plot_instance(instance), use_container_width=True)

            # Botón de descarga CSV
            buf = io.StringIO()
            import pandas as pd
            import numpy as np
            import tempfile, os
            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
                tmp_path = tmp.name
            write_instance_csv(instance, tmp_path)
            with open(tmp_path, "rb") as f:
                st.download_button(
                    "⬇️ Descargar esta instancia como CSV",
                    f.read(),
                    file_name=f"{instance.name}.csv",
                    mime="text/csv",
                )
            os.unlink(tmp_path)

        except ValueError as e:
            st.error(f"Error: {e}")

# ---------------------------------------------------------------------------
# Tab 3: Subir CSV
# ---------------------------------------------------------------------------
with tab3:
    st.subheader("Subir datos desde CSV")
    st.markdown(
        "El CSV debe tener las columnas: `node`, `x`, `y`, `demand`, `capacity`, `name`.\n\n"
        "El nodo 0 es el depósito (`demand=0`). La columna `capacity` debe tener "
        "el mismo valor en todas las filas."
    )

    template_df = csv_template()
    csv_bytes = template_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Descargar plantilla CSV", csv_bytes, "plantilla_vrp.csv", "text/csv")

    uploaded = st.file_uploader("Sube tu archivo CSV", type=["csv"])
    if uploaded:
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="wb") as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name
        try:
            instance = read_instance_csv(tmp_path)
            st.session_state["instance"] = instance
            st.success(f"CSV cargado: '{instance.name}' — {instance.n_customers} clientes, Q={instance.capacity}")

            col_info3, col_plot3 = st.columns([1, 2])
            with col_info3:
                st.json(instance.summary())
            with col_plot3:
                st.plotly_chart(plot_instance(instance), use_container_width=True)
        except (ValueError, FileNotFoundError) as e:
            st.error(f"Error de validación: {e}")
        finally:
            os.unlink(tmp_path)

# ---------------------------------------------------------------------------
# Instancia activa en sidebar
# ---------------------------------------------------------------------------
if "instance" in st.session_state:
    inst = st.session_state["instance"]
    with st.sidebar:
        st.success(f"**Instancia activa:**\n{inst.name}\n\n{inst.n_customers} clientes | Q={inst.capacity}")
