import streamlit as st

from app.algorithms import ALL_ALGORITHMS
from app.utils.visualization import bar_comparison, comparison_table, plot_routes

st.set_page_config(page_title="Comparar algoritmos", layout="wide")
st.title("📊 Comparar algoritmos")

if "instance" not in st.session_state:
    st.warning("Primero carga una instancia en la página **Instancias**.")
    st.stop()

instance = st.session_state["instance"]
st.info(
    f"Instancia activa: **{instance.name}** | "
    f"{instance.n_customers} clientes | Q={instance.capacity}"
)

selected_algos = st.multiselect(
    "Selecciona los algoritmos a comparar:",
    list(ALL_ALGORITHMS.keys()),
    default=list(ALL_ALGORITHMS.keys()),
)

if st.button("▶️ Comparar todos"):
    solutions = []
    with st.spinner("Ejecutando algoritmos..."):
        for name in selected_algos:
            sol = ALL_ALGORITHMS[name](instance)
            solutions.append(sol)
    st.session_state["comparison_solutions"] = solutions

if "comparison_solutions" in st.session_state:
    solutions = st.session_state["comparison_solutions"]

    st.subheader("Tabla comparativa")
    df = comparison_table(solutions)
    st.dataframe(df, use_container_width=True)

    st.subheader("Gráfico comparativo")
    st.plotly_chart(bar_comparison(solutions), use_container_width=True)

    st.subheader("Ranking de desempeño")
    ranked = sorted(solutions, key=lambda s: s.total_distance())
    for i, sol in enumerate(ranked):
        medal = ["🥇", "🥈", "🥉"][i] if i < 3 else f"{i+1}."
        st.write(
            f"{medal} **{sol.algorithm_name}** — "
            f"Distancia: {sol.total_distance():.2f} | "
            f"Vehículos: {sol.vehicles_used()} | "
            f"Tiempo: {sol.computation_time:.4f}s"
        )

    st.subheader("Visualización de rutas por algoritmo")
    cols = st.columns(min(len(solutions), 2))
    for i, sol in enumerate(solutions):
        with cols[i % 2]:
            st.plotly_chart(plot_routes(sol), use_container_width=True)
