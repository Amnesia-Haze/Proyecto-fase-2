import streamlit as st

st.set_page_config(
    page_title="VRP Optimizer — Gestión de Operaciones",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🚚 VRP Optimizer")
st.subheader("Herramienta educativa para ruteo y optimización logística")

st.markdown("""
Bienvenido al optimizador de rutas para el curso de **Gestión de Operaciones**.

Esta aplicación te permite:

| Módulo | Descripción |
|--------|-------------|
| 📦 **Instancias** | Cargar, generar o subir instancias VRP |
| ⚙️ **Resolver** | Aplicar heurísticas y solvers a una instancia |
| 📊 **Comparar** | Comparar el desempeño de múltiples algoritmos |

---
**¿Cómo empezar?**
Usa el menú lateral para navegar entre las páginas.
""")

st.info("💡 Selecciona una página en el panel izquierdo para comenzar.")
