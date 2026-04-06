import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pinn_solver import PINNConfig, train_pinn


def test_pinn_smoke():
    cfg = PINNConfig(
        epochs=2,
        n_residual=20,
        n_boundary=12,
        n_initial=12,
        hidden_layers=1,
        hidden_width=8,
        print_every=1,
        eval_grid_size=5,
        eval_times=(0.0, 0.1),
        use_hard_constraints=True,
    )
    _, history, metrics, _ = train_pinn(cfg)
    assert len(history["epoch"]) == 2
    assert len(metrics) == 2
