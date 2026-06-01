from . import nearest_neighbor, clarke_wright, sweep

AVAILABLE = {
    "Nearest Neighbor": nearest_neighbor.solve,
    "Clarke & Wright Savings": clarke_wright.solve,
    "Sweep Algorithm": sweep.solve,
}
