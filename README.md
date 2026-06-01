# VRP Optimizer — Gestión de Operaciones

Aplicación educativa para resolver y visualizar problemas de **Vehicle Routing Problem (VRP)** usando heurísticas y solvers matemáticos.

## Requisitos

- Python 3.11+
- pip

## Instalación

```bash
pip install -r requirements.txt
```

## Ejecución

```bash
streamlit run app/main.py
```

La aplicación abrirá en el navegador en `http://localhost:8501`.

## Estructura del proyecto

```
vrp-optimizer/
├── app/
│   ├── main.py                    # Página principal Streamlit
│   ├── pages/
│   │   ├── 1_Instancias.py        # Cargar / generar / subir instancias
│   │   ├── 2_Resolver.py          # Resolver con un algoritmo
│   │   └── 3_Comparar.py          # Comparar múltiples algoritmos
│   ├── algorithms/
│   │   ├── heuristics/
│   │   │   ├── nearest_neighbor.py
│   │   │   ├── clarke_wright.py
│   │   │   └── sweep.py
│   │   └── solvers/               # (OR-Tools, PuLP — fase siguiente)
│   ├── models/
│   │   └── vrp_instance.py        # Modelos de datos: VRPInstance, VRPSolution
│   └── utils/
│       ├── data_loader.py         # Carga y generación de instancias
│       ├── visualization.py       # Gráficos Plotly
│       └── preloaded_instances.py # Instancias de ejemplo por dificultad
├── data/
│   ├── instances/                 # Instancias guardadas (small/medium/large)
│   └── templates/
│       └── plantilla_vrp.csv      # Plantilla descargable
├── tests/
├── docs/
├── requirements.txt
└── README.md
```

## Algoritmos implementados

| Tipo | Algoritmo |
|------|-----------|
| Heurística | Nearest Neighbor |
| Heurística | Clarke & Wright Savings |
| Heurística | Sweep Algorithm |
| Solver exacto | OR-Tools *(próxima fase)* |
| Solver exacto | PuLP / CBC *(próxima fase)* |
| Metaheurística | Simulated Annealing *(fase futura)* |
