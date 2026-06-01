from app.models.instances import generate_cvrp_instance

PRELOADED = {
    "Pequeña — Distribución urbana (8 clientes)": generate_cvrp_instance(
        n_customers=8, capacity=30, demand_range=(1, 15), seed=1,
        name="Distribución urbana (8 clientes)",
    ),
    "Pequeña — Entrega última milla (10 clientes)": generate_cvrp_instance(
        n_customers=10, capacity=40, demand_range=(1, 15), seed=2,
        name="Entrega última milla (10 clientes)",
    ),
    "Mediana — Recolección de residuos (20 clientes)": generate_cvrp_instance(
        n_customers=20, capacity=50, demand_range=(1, 20), seed=3,
        name="Recolección de residuos (20 clientes)",
    ),
    "Mediana — Transporte de mercancías (25 clientes)": generate_cvrp_instance(
        n_customers=25, capacity=60, demand_range=(1, 20), seed=4,
        name="Transporte de mercancías (25 clientes)",
    ),
    "Grande — Red logística urbana (50 clientes)": generate_cvrp_instance(
        n_customers=50, capacity=80, demand_range=(1, 20), seed=5,
        name="Red logística urbana (50 clientes)",
    ),
    "Grande — Distribución regional (80 clientes)": generate_cvrp_instance(
        n_customers=80, capacity=100, demand_range=(1, 20), seed=6,
        name="Distribución regional (80 clientes)",
    ),
}
