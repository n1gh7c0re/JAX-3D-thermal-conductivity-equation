import sys
import os
import csv
from datetime import datetime

# === Добавляем корень проекта в PYTHONPATH ===
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import jax.numpy as jnp
from src.fdm_solver import solve_fdm_3d


def analytical_solution(X, Y, Z, t, Lx=1.0, Ly=1.0, Lz=1.0, alpha=1.0):
    """Точное аналитическое решение"""
    k = jnp.pi**2 * (1.0/Lx**2 + 1.0/Ly**2 + 1.0/Lz**2)
    return jnp.exp(-alpha * k * t) * jnp.sin(jnp.pi * X / Lx) * jnp.sin(jnp.pi * Y / Ly) * jnp.sin(jnp.pi * Z / Lz)


def compute_errors(Nx: int, Ny: int, Nz: int, T: float = 0.1, alpha: float = 1.0,
                   Lx: float = 1.0, Ly: float = 1.0, Lz: float = 1.0):
    """L2 и L∞ ошибки + параметры"""
    dx = Lx / (Nx - 1)
    dt = 0.1 * (dx**2) / (6.0 * alpha)

    u_num = solve_fdm_3d(Nx, Ny, Nz, dt, T, alpha, Lx, Ly, Lz)

    x = jnp.linspace(0.0, Lx, Nx)
    y = jnp.linspace(0.0, Ly, Ny)
    z = jnp.linspace(0.0, Lz, Nz)
    X, Y, Z = jnp.meshgrid(x, y, z, indexing='ij')
    u_ana = analytical_solution(X, Y, Z, T, Lx, Ly, Lz, alpha)

    err_l2 = jnp.sqrt(jnp.mean((u_num - u_ana)**2))
    err_linf = jnp.max(jnp.abs(u_num - u_ana))
    return err_l2, err_linf, dt, Nx * Ny * Nz


if __name__ == "__main__":
    print("=== FDM Solver 3D Heat Equation (JAX) — тесты ===")

    # 1. Проверка граничных условий
    print("\n1. Проверка граничных условий (Nx=Ny=Nz=11, T=0.05)")
    u_test = solve_fdm_3d(11, 11, 11, dt=0.0001, T=0.05)
    assert jnp.allclose(u_test[0, :, :], 0.0), "BC x=0"
    assert jnp.allclose(u_test[-1, :, :], 0.0), "BC x=Lx"
    assert jnp.allclose(u_test[:, 0, :], 0.0), "BC y=0"
    assert jnp.allclose(u_test[:, -1, :], 0.0), "BC y=Ly"
    assert jnp.allclose(u_test[:, :, 0], 0.0), "BC z=0"
    assert jnp.allclose(u_test[:, :, -1], 0.0), "BC z=Lz"
    print("✅ Все граничные условия = 0.0")

    # 2. Таблица сходимости + запись в CSV
    print("\n2. Таблица сходимости")
    print("Nx\tL2 error\tL∞ error\tdt\t\tN_points")
    print("-" * 75)

    # Создаём папку results, если её нет
    os.makedirs("results", exist_ok=True)
    csv_path = "results/metrics.csv"

    # Заголовки для CSV
    headers = ["Nx", "L2_error", "Linf_error", "dt", "N_points", "timestamp"]

    # Открываем файл для записи (создаём заново при каждом запуске)
    with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        grid_sizes = [11, 21, 31, 41]
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        for N in grid_sizes:
            err_l2, err_linf, dt_val, n_points = compute_errors(N, N, N)
            
            # Вывод на экран
            print(f"{N}\t{err_l2:.2e}\t{err_linf:.2e}\t{dt_val:.2e}\t{n_points}")

            # Запись в CSV
            writer.writerow([
                N,
                f"{err_l2:.2e}",
                f"{err_linf:.2e}",
                f"{dt_val:.2e}",
                n_points,
                timestamp
            ])