from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax.numpy as jnp
import matplotlib.pyplot as plt

from src.pinn_solver import PINNConfig, build_eval_grid, exact_solution, predict, save_training_outputs, train_pinn


def _reshape_to_grid(values, n):
    return values.reshape(n, n, n)


def plot_loss_curves(history, output_dir: Path):
    plt.figure(figsize=(8, 5))
    plt.plot(history["epoch"], history["loss_total"], label="total")
    plt.plot(history["epoch"], history["loss_pde"], label="pde")
    if max(history["loss_bc"]) > 0:
        plt.plot(history["epoch"], history["loss_bc"], label="bc")
    if max(history["loss_ic"]) > 0:
        plt.plot(history["epoch"], history["loss_ic"], label="ic")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("PINN training losses")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curves.png", dpi=160)
    plt.close()


def plot_learning_rate(history, output_dir: Path):
    plt.figure(figsize=(8, 4))
    plt.plot(history["epoch"], history["learning_rate"])
    plt.xlabel("Epoch")
    plt.ylabel("Learning rate")
    plt.title("Learning-rate schedule")
    plt.tight_layout()
    plt.savefig(output_dir / "learning_rate.png", dpi=160)
    plt.close()


def plot_slice_comparison(params, cfg: PINNConfig, time_value: float, output_dir: Path):
    n = cfg.eval_grid_size
    pts = build_eval_grid(cfg, time_value, n=n)
    u_pred = predict(params, pts, cfg).reshape(-1)
    u_exact = exact_solution(pts, alpha=cfg.alpha, Lx=cfg.Lx, Ly=cfg.Ly, Lz=cfg.Lz).reshape(-1)
    u_err = jnp.abs(u_pred - u_exact)

    pred_3d = _reshape_to_grid(u_pred, n)
    exact_3d = _reshape_to_grid(u_exact, n)
    err_3d = _reshape_to_grid(u_err, n)

    z_idx = n // 2
    pred_slice = pred_3d[:, :, z_idx]
    exact_slice = exact_3d[:, :, z_idx]
    err_slice = err_3d[:, :, z_idx]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    im0 = axes[0].imshow(exact_slice.T, origin="lower", aspect="auto")
    axes[0].set_title(f"Exact, t={time_value:.3f}")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(pred_slice.T, origin="lower", aspect="auto")
    axes[1].set_title(f"PINN, t={time_value:.3f}")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    im2 = axes[2].imshow(err_slice.T, origin="lower", aspect="auto")
    axes[2].set_title("|Error|")
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    for ax in axes:
        ax.set_xlabel("x index")
        ax.set_ylabel("y index")

    plt.tight_layout()
    fig.savefig(output_dir / f"slice_comparison_t_{str(time_value).replace('.', '_')}.png", dpi=160)
    plt.close(fig)


def save_metrics_table(metrics_by_time, output_dir: Path):
    lines = ["time,l2,linf,rel_l2"]
    for m in metrics_by_time:
        lines.append(f"{m['time']},{m['l2']},{m['linf']},{m['rel_l2']}")
    (output_dir / "metrics.csv").write_text("\n".join(lines), encoding="utf-8")


def main():
    cfg = PINNConfig(
        alpha=1.0,
        T=0.1,
        hidden_width=96,
        hidden_layers=5,
        n_residual=12000,
        n_boundary=2000,
        n_initial=2000,
        learning_rate=1e-3,
        min_learning_rate=1e-4,
        epochs=8000,
        print_every=250,
        use_hard_constraints=True,
        pretrain_ic_bc_epochs=0,
        eval_grid_size=21,
        eval_times=(0.0, 0.02, 0.05, 0.1),
        seed=42,
    )

    output_dir = Path("results/pinn")
    output_dir.mkdir(parents=True, exist_ok=True)

    params, history, metrics_by_time, cfg = train_pinn(cfg)
    save_training_outputs(output_dir, params, history, metrics_by_time, cfg)
    save_metrics_table(metrics_by_time, output_dir)
    plot_loss_curves(history, output_dir)
    plot_learning_rate(history, output_dir)

    for t in cfg.eval_times:
        plot_slice_comparison(params, cfg, t, output_dir)

    print("\n=== PINN validation metrics ===")
    for m in metrics_by_time:
        print(f"t={m['time']:.3f} | L2={m['l2']:.3e} | Linf={m['linf']:.3e} | rel_L2={m['rel_l2']:.3e}")

    print(f"\nSaved outputs to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
