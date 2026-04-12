from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pinn_solver import PINNConfig, save_training_outputs, train_pinn
from experiments.run_pinn_validation import (
    plot_learning_rate,
    plot_loss_curves,
    plot_slice_comparison,
    save_metrics_table,
)


def build_demo_config() -> PINNConfig:
    """Tiny PINN configuration for a quick smoke-style demo run."""
    return PINNConfig(
        alpha=1.0,
        T=0.1,
        hidden_width=96,
        hidden_layers=5,
        n_residual=1000,
        n_boundary=200,
        n_initial=200,
        learning_rate=1e-3,
        min_learning_rate=1e-4,
        epochs=50,
        print_every=10,
        use_hard_constraints=True,
        pretrain_ic_bc_epochs=0,
        eval_grid_size=21,
        eval_times=(0.0, 0.02, 0.05, 0.1),
        seed=42,
    )


def main() -> None:
    cfg = build_demo_config()
    output_dir = Path("results/pinn_demo")
    output_dir.mkdir(parents=True, exist_ok=True)

    params, history, metrics_by_time, cfg = train_pinn(cfg)
    save_training_outputs(output_dir, params, history, metrics_by_time, cfg)
    save_metrics_table(metrics_by_time, output_dir)
    plot_loss_curves(history, output_dir)
    plot_learning_rate(history, output_dir)

    for t in cfg.eval_times:
        plot_slice_comparison(params, cfg, t, output_dir)

    print("\n=== PINN demo validation metrics ===")
    for m in metrics_by_time:
        print(
            f"t={m['time']:.3f} | L2={m['l2']:.3e} | "
            f"Linf={m['linf']:.3e} | rel_L2={m['rel_l2']:.3e}"
        )

    print(f"\nSaved demo outputs to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
