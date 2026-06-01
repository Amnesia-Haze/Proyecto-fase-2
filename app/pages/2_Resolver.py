import streamlit as st

from app.algorithms import ALL_ALGORITHMS
from app.utils.visualization import plot_routes

st.set_page_config(page_title="Resolver CVRP", layout="wide")
st.title("⚙️ Resolver instancia")

if "instance" not in st.session_state:
    st.warning("Primero carga una instancia en la página **Instancias**.")
    st.stop()

instance = st.session_state["instance"]
st.info(
    f"Instancia activa: **{instance.name}** | "
    f"{instance.n_customers} clientes | Q={instance.capacity}"
)

st.subheader("Selecciona un algoritmo")
algo_name = st.selectbox("Algoritmo:", list(ALL_ALGORITHMS.keys()))

if st.button("▶️ Resolver"):
    with st.spinner("Resolviendo..."):
        solution = ALL_ALGORITHMS[algo_name](instance)

    if "solutions" not in st.session_state:
        st.session_state["solutions"] = []
    st.session_state["solutions"].append(solution)
    st.session_state["last_solution"] = solution
    st.success(f"Resuelto en {solution.computation_time:.4f}s")

if "last_solution" in st.session_state:
    sol = st.session_state["last_solution"]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Distancia total", f"{sol.total_distance():.2f}")
    col2.metric("Vehículos usados", sol.vehicles_used())
    col3.metric("Tiempo (s)", f"{sol.computation_time:.4f}")
    utils = sol.capacity_utilization()
    col4.metric("Utilización prom. (%)", f"{sum(utils)/len(utils):.1f}" if utils else "—")

    st.plotly_chart(plot_routes(sol), use_container_width=True)

    with st.expander("Ver detalle de rutas"):
        for i, route in enumerate(sol.routes):
            if not route:
                continue
            demand = sum(instance.demands[c] for c in route)
            pct = demand / instance.capacity * 100
            nodes_str = " → ".join(str(c) for c in route)
            st.write(
                f"**Ruta {i+1}:** {nodes_str} | "
                f"Demanda: {demand:.1f} / {instance.capacity} ({pct:.1f}%)"
            )
