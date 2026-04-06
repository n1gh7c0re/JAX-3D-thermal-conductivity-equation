from __future__ import annotations
import sys
from pathlib import Path
import csv
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax.numpy as jnp
import matplotlib.pyplot as plt
from src.fdm_solver import solve_fdm_3d


def analytical_solution(X, Y, Z, t, Lx=1.0, Ly=1.0, Lz=1.0, alpha=1.0):
    """Точное аналитическое решение"""
    k = jnp.pi**2 * (1.0/Lx**2 + 1.0/Ly**2 + 1.0/Lz**2)
    return jnp.exp(-alpha * k * t) * jnp.sin(jnp.pi * X / Lx) * jnp.sin(jnp.pi * Y / Ly) * jnp.sin(jnp.pi * Z / Lz)


def compute_metrics(Nx: int, Ny: int, Nz: int, T: float = 0.1, alpha: float = 1.0):
    """Вычисляет метрики для заданной сетки"""
    Lx = Ly = Lz = 1.0
    
    dx = Lx / (Nx - 1)
    dy = Ly / (Ny - 1)
    dz = Lz / (Nz - 1)
    
    # Безопасный шаг по времени (CFL = 0.1)
    dt = 0.1 * min(dx, dy, dz)**2 / (6.0 * alpha)

    u_num = solve_fdm_3d(Nx, Ny, Nz, dt, T, alpha, Lx, Ly, Lz)

    x = jnp.linspace(0.0, Lx, Nx)
    y = jnp.linspace(0.0, Ly, Ny)
    z = jnp.linspace(0.0, Lz, Nz)
    X, Y, Z = jnp.meshgrid(x, y, z, indexing='ij')
    u_exact = analytical_solution(X, Y, Z, T, Lx, Ly, Lz, alpha)

    err_l2 = float(jnp.sqrt(jnp.mean((u_num - u_exact)**2)))
    err_linf = float(jnp.max(jnp.abs(u_num - u_exact)))
    rel_l2 = err_l2 / (float(jnp.max(jnp.abs(u_exact))) + 1e-8)

    return {
        "Nx": Nx,
        "L2": err_l2,
        "Linf": err_linf,
        "rel_L2": rel_l2,
        "dt": dt,
        "N_points": Nx * Ny * Nz,
        "u_num": u_num,
        "u_exact": u_exact
    }


def plot_slice_comparison(u_num, u_exact, time_value: float, output_dir: Path, grid_size: int):
    """Построение среза сравнения по z = 0.5"""
    z_idx = grid_size // 2
    pred_slice = u_num[:, :, z_idx]
    exact_slice = u_exact[:, :, z_idx]
    err_slice = jnp.abs(u_num - u_exact)[:, :, z_idx]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    im0 = axes[0].imshow(exact_slice.T, origin="lower", cmap="viridis")
    axes[0].set_title(f"Exact Solution, t={time_value:.3f}")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(pred_slice.T, origin="lower", cmap="viridis")
    axes[1].set_title(f"FDM Solution, t={time_value:.3f}")
    plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(err_slice.T, origin="lower", cmap="Reds")
    axes[2].set_title(f"|Error|, t={time_value:.3f}")
    plt.colorbar(im2, ax=axes[2])

    for ax in axes:
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    plt.tight_layout()
    fig.savefig(output_dir / f"fdm_slice_t_{str(time_value).replace('.', '_')}.png", 
                dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    output_dir = Path("results/fdm")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== FDM Validation Script ===")
    print("Запуск полноценной валидации сеточного метода...\n")

    grid_sizes = [11, 21, 31, 41]
    metrics_list = []
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print("Nx\tL2 error\tLinf error\tdt\t\tN_points")
    print("-" * 80)

    for N in grid_sizes:
        metrics = compute_metrics(N, N, N, T=0.1)
        metrics_list.append(metrics)

        print(f"{N}\t{metrics['L2']:.2e}\t{metrics['Linf']:.2e}\t{metrics['dt']:.2e}\t{metrics['N_points']}")

    # Сохранение метрик
    csv_path = output_dir / "metrics.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Nx", "L2", "Linf", "rel_L2", "dt", "N_points", "timestamp"])
        for m in metrics_list:
            writer.writerow([
                m["Nx"],
                f"{m['L2']:.2e}",
                f"{m['Linf']:.2e}",
                f"{m['rel_L2']:.2e}",
                f"{m['dt']:.2e}",
                m["N_points"],
                timestamp
            ])

    # Сохранение решений для самой точной сетки
    finest = metrics_list[-1]
    jnp.save(output_dir / "fdm_u_num_N41.npy", finest["u_num"])
    jnp.save(output_dir / "fdm_u_exact_N41.npy", finest["u_exact"])

    # Визуализация
    plot_slice_comparison(finest["u_num"], finest["u_exact"], 
                         time_value=0.1, output_dir=output_dir, grid_size=41)


if __name__ == "__main__":
    main()
