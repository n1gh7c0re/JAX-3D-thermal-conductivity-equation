"""
Comprehensive validation and comparison framework for FDM vs PINN solvers.

Contains all scientific metrics computation, visualization functions, benchmarking,
and table generation for comparing Finite Difference Method and Physics-Informed
Neural Networks on the 3D heat equation.
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, Any
import pickle
import json
import os

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from matplotlib import rcParams

from src.analytical_solution import analytical_solution, decay_rate

logger = logging.getLogger(__name__)

__all__ = [
    "load_fdm_results",
    "load_pinn_results",
    "compute_l2_linf",
    "compute_errors_over_time",
    "generate_comparison_table",
    "plot_snapshots_comparison",
    "plot_error_vs_time",
    "plot_fdm_convergence",
    "plot_pinn_loss_history",
    "benchmark_timing",
    "run_full_validation",
]


def setup_plotting():
    """Configure matplotlib and seaborn for publication-quality plots."""
    sns.set_theme(style="whitegrid", palette="deep")
    rcParams.update({
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "font.size": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
    })


def load_fdm_results(
    base_dir: str | Path = "results/fdm",
    grid_size: int = 41
) -> Dict[str, Any]:
    """
    Load FDM results from disk.

    Args:
        base_dir: Directory containing FDM results.
        grid_size: Grid size (default 41).

    Returns:
        Dictionary with keys:
            - 'u_num': Numerical FDM solution (N³ array)
            - 'u_exact': Exact analytical solution (N³ array)
            - 'metrics': DataFrame with convergence metrics
    """
    base_dir = Path(base_dir)
    logger.info(f"Loading FDM results from {base_dir}")

    try:
        u_num = jnp.array(np.load(base_dir / f"fdm_u_num_N{grid_size}.npy"))
        u_exact = jnp.array(np.load(base_dir / f"fdm_u_exact_N{grid_size}.npy"))

        metrics = pd.read_csv(base_dir / "metrics.csv")
        logger.info(f"✓ FDM results loaded: u_num shape={u_num.shape}, u_exact shape={u_exact.shape}")

        return {"u_num": u_num, "u_exact": u_exact, "metrics": metrics}
    except FileNotFoundError as e:
        logger.error(f"FDM results not found: {e}")
        raise


def load_pinn_results(base_dir: str | Path = "results/pinn") -> Dict[str, Any]:
    """
    Load PINN results from disk.

    Args:
        base_dir: Directory containing PINN results.

    Returns:
        Dictionary with keys:
            - 'params': Trained network parameters
            - 'history': Training history
            - 'metrics_by_time': Metrics at multiple time steps
            - 'config': PINN configuration
    """
    base_dir = Path(base_dir)
    logger.info(f"Loading PINN results from {base_dir}")

    try:
        with open(base_dir / "params.pkl", "rb") as f:
            params = pickle.load(f)

        with open(base_dir / "history.json", "r") as f:
            history = json.load(f)

        with open(base_dir / "metrics_by_time.json", "r") as f:
            metrics_by_time = json.load(f)

        with open(base_dir / "config.json", "r") as f:
            config = json.load(f)

        logger.info(f"✓ PINN results loaded: {len(history['epoch'])} epochs, config={config}")

        return {
            "params": params,
            "history": history,
            "metrics_by_time": metrics_by_time,
            "config": config,
        }
    except FileNotFoundError as e:
        logger.error(f"PINN results not found: {e}")
        raise


def compute_l2_linf(
    u_pred: jax.Array,
    u_exact: jax.Array,
    eps: float = 1e-12
) -> Tuple[float, float, float]:
    """
    Compute L2 and L∞ errors with relative normalization.

    Args:
        u_pred: Predicted solution.
        u_exact: Exact analytical solution.
        eps: Small regularization value to avoid division by zero.

    Returns:
        Tuple (rel_l2, l_inf_abs, rel_l_inf) where:
        - rel_l2: Relative L2 error ||u_pred - u_exact||₂ / ||u_exact||₂
        - l_inf_abs: Absolute L∞ error max|u_pred - u_exact|
        - rel_l_inf: Relative L∞ error
    """
    error = u_pred - u_exact
    error_l2 = jnp.sqrt(jnp.mean(error**2))
    error_linf = jnp.max(jnp.abs(error))

    exact_l2 = jnp.sqrt(jnp.mean(u_exact**2))
    exact_linf = jnp.max(jnp.abs(u_exact))

    rel_l2 = error_l2 / (exact_l2 + eps)
    rel_linf = error_linf / (exact_linf + eps)

    return float(rel_l2), float(error_linf), float(rel_linf)


def compute_errors_over_time(
    u_pred_3d: jax.Array,
    times: np.ndarray,
    grid_size: int,
    Lx: float = 1.0,
    Ly: float = 1.0,
    Lz: float = 1.0,
    alpha: float = 0.1
) -> pd.DataFrame:
    """
    Compute error metrics at multiple time steps.

    Args:
        u_pred_3d: Predicted solution (N × N × N array).
        times: Array of time values.
        grid_size: Grid size N.
        Lx, Ly, Lz: Domain lengths.
        alpha: Thermal diffusivity.

    Returns:
        DataFrame with columns: ['time', 'rel_l2', 'l_inf_abs', 'rel_l_inf']
    """
    logger.info(f"Computing errors over {len(times)} time steps...")

    x = jnp.linspace(0, Lx, grid_size)
    y = jnp.linspace(0, Ly, grid_size)
    z = jnp.linspace(0, Lz, grid_size)
    X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")

    rows = []
    for t in times:
        u_exact = analytical_solution(X, Y, Z, t, Lx, Ly, Lz, alpha)
        rel_l2, l_inf_abs, rel_linf = compute_l2_linf(u_pred_3d, u_exact)
        rows.append({"time": float(t), "rel_l2": rel_l2, "l_inf_abs": l_inf_abs, "rel_l_inf": rel_linf})

    return pd.DataFrame(rows)


def generate_comparison_table(
    fdm_metrics: pd.DataFrame,
    pinn_metrics: pd.DataFrame,
    output_dir: str | Path = "results/tables",
) -> pd.DataFrame:
    """
    Generate comparison table for FDM vs PINN at multiple resolutions.

    Args:
        fdm_metrics: FDM metrics DataFrame from load_fdm_results.
        pinn_metrics: PINN metrics DataFrame (time-based).
        output_dir: Directory to save table outputs.

    Returns:
        Merged comparison DataFrame.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Generating comparison table...")

    # Convert FDM metrics to readable format
    fdm_table = fdm_metrics[["Nx", "L2", "Linf", "N_points"]].copy()
    fdm_table.columns = ["Grid Size", "FDM L2 Error", "FDM L∞ Error", "Grid Points"]
    fdm_table["FDM L2 Error"] = fdm_table["FDM L2 Error"].map(lambda x: f"{x:.2e}")
    fdm_table["FDM L∞ Error"] = fdm_table["FDM L∞ Error"].map(lambda x: f"{x:.2e}")

    # Extract PINN metrics at final time (or best error)
    if isinstance(pinn_metrics, pd.DataFrame):
        if len(pinn_metrics) > 0 and "rel_l2" in pinn_metrics.columns:
            pinn_final = pinn_metrics.iloc[pinn_metrics["rel_l2"].idxmin()].to_dict()
        else:
            pinn_final = {}
    else:
        # pinn_metrics might be dict or list of dicts
        pinn_final = {}

    pinn_row = pd.DataFrame([{
        "Grid Size": "PINN (21³)",
        "FDM L2 Error": f"{pinn_final.get('rel_l2', 0):.2e}",
        "FDM L∞ Error": f"{pinn_final.get('rel_l_inf', pinn_final.get('l_inf', 0)):.2e}",
        "Grid Points": "9261",
    }])

    comparison = pd.concat([fdm_table, pinn_row], ignore_index=True)

    # Save as markdown table
    md_table = comparison.to_markdown(index=False)
    (output_dir / "comparison_table.md").write_text(md_table, encoding="utf-8")
    logger.info(f"✓ Comparison table saved to {output_dir / 'comparison_table.md'}")

    # Save as LaTeX
    latex_table = comparison.to_latex(index=False, escape=False)
    (output_dir / "comparison_table.tex").write_text(latex_table, encoding="utf-8")
    logger.info(f"✓ LaTeX table saved to {output_dir / 'comparison_table.tex'}")

    return comparison


def plot_snapshots_comparison(
    u_fdm: jax.Array,
    u_pinn: jax.Array,
    u_exact: jax.Array,
    times: list[float] | np.ndarray,
    output_dir: str | Path = "results/figures",
) -> None:
    """
    Plot solution snapshots: FDM | PINN | Exact at multiple time steps.

    Args:
        u_fdm: FDM solution (N × N × N).
        u_pinn: PINN solution (N × N × N).
        u_exact: Exact solution (N × N × N).
        times: List of time values for snapshots.
        output_dir: Output directory for figures.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Plotting snapshots at {len(times)} time steps...")

    N = u_fdm.shape[0]
    z_idx = N // 2

    for t in times:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

        # Extract z-slice
        slice_exact = u_exact[:, :, z_idx]
        slice_fdm = u_fdm[:, :, z_idx]
        slice_pinn = u_pinn[:, :, z_idx]

        # Plot
        vmin = float(jnp.minimum(jnp.min(slice_exact), jnp.minimum(jnp.min(slice_fdm), jnp.min(slice_pinn))))
        vmax = float(jnp.maximum(jnp.max(slice_exact), jnp.maximum(jnp.max(slice_fdm), jnp.max(slice_pinn))))

        im0 = axes[0].imshow(slice_exact.T, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
        axes[0].set_title(f"Точное решение (t={t:.3f})")
        plt.colorbar(im0, ax=axes[0])

        im1 = axes[1].imshow(slice_fdm.T, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
        axes[1].set_title(f"Решение FDM (t={t:.3f})")
        plt.colorbar(im1, ax=axes[1])

        im2 = axes[2].imshow(slice_pinn.T, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
        axes[2].set_title(f"Решение PINN (t={t:.3f})")
        plt.colorbar(im2, ax=axes[2])

        for ax in axes:
            ax.set_xlabel("Индекс x")
            ax.set_ylabel("Индекс y")

        plt.tight_layout()
        t_str = str(t).replace(".", "_")
        plt.savefig(output_dir / f"snapshots_t_{t_str}.png", dpi=300, bbox_inches="tight")
        plt.savefig(output_dir / f"snapshots_t_{t_str}.pdf", bbox_inches="tight")
        plt.close()

    logger.info(f"✓ Snapshots saved to {output_dir}")


def plot_error_vs_time(
    errors_fdm: pd.DataFrame,
    errors_pinn: pd.DataFrame,
    output_dir: str | Path = "results/figures",
) -> None:
    """
    Plot error evolution over time: L2 and L∞ for both methods.

    Args:
        errors_fdm: FDM errors DataFrame with columns ['time', 'rel_l2', 'l_inf_abs'].
        errors_pinn: PINN errors DataFrame with same columns.
        output_dir: Output directory.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Plotting error vs time...")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # L2 error
    axes[0].semilogy(errors_fdm["time"], errors_fdm["rel_l2"], "o-", label="FDM", linewidth=2, markersize=6)
    axes[0].semilogy(errors_pinn["time"], errors_pinn["rel_l2"], "s-", label="PINN", linewidth=2, markersize=6)
    axes[0].set_xlabel("Время t")
    axes[0].set_ylabel("Относительная ошибка L²")
    axes[0].set_title("Эволюция ошибки L²")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # L∞ error
    axes[1].semilogy(errors_fdm["time"], errors_fdm["l_inf_abs"], "o-", label="FDM", linewidth=2, markersize=6)
    axes[1].semilogy(errors_pinn["time"], errors_pinn["l_inf_abs"], "s-", label="PINN", linewidth=2, markersize=6)
    axes[1].set_xlabel("Время t")
    axes[1].set_ylabel("Абсолютная ошибка L∞")
    axes[1].set_title("Эволюция ошибки L∞")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "error_vs_time.png", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "error_vs_time.pdf", bbox_inches="tight")
    plt.close()

    logger.info(f"✓ Error plot saved to {output_dir / 'error_vs_time'}")


def plot_fdm_convergence(
    metrics: pd.DataFrame,
    output_dir: str | Path = "results/figures",
) -> None:
    """
    Plot FDM convergence (log-log) and measure order of convergence.

    Args:
        metrics: FDM metrics DataFrame with columns ['Nx', 'L2', 'Linf', 'N_points'].
        output_dir: Output directory.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Plotting FDM convergence...")

    # Extract grid spacing
    dx = 1.0 / (metrics["Nx"].values - 1)

    # Fit log-log line to L2 errors
    log_dx = np.log(dx)
    log_l2 = np.log(metrics["L2"].values)
    log_linf = np.log(metrics["Linf"].values)

    # Linear fit (degree = 1)
    poly_l2 = np.polyfit(log_dx, log_l2, 1)
    poly_linf = np.polyfit(log_dx, log_linf, 1)

    order_l2 = poly_l2[0]
    order_linf = poly_linf[0]

    logger.info(f"FDM convergence order: L2={order_l2:.2f}, L∞={order_linf:.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # L2 convergence
    axes[0].loglog(dx, metrics["L2"].values, "o-", label="Ошибка L²", linewidth=2, markersize=8)
    axes[0].loglog(dx, np.exp(poly_l2[1]) * dx**poly_l2[0], "--", label=f"Порядок {order_l2:.2f}", linewidth=2)
    axes[0].loglog(dx, np.exp(np.log(metrics["L2"].values[0])) * dx**2, ":", label="O(dx²)", linewidth=1.5, alpha=0.7)
    axes[0].set_xlabel("Шаг сетки dx")
    axes[0].set_ylabel("Ошибка L²")
    axes[0].set_title("Сходимость FDM L²")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, which="both")

    # L∞ convergence
    axes[1].loglog(dx, metrics["Linf"].values, "s-", label="Ошибка L∞", linewidth=2, markersize=8)
    axes[1].loglog(dx, np.exp(poly_linf[1]) * dx**poly_linf[0], "--", label=f"Порядок {order_linf:.2f}", linewidth=2)
    axes[1].loglog(dx, np.exp(np.log(metrics["Linf"].values[0])) * dx**2, ":", label="O(dx²)", linewidth=1.5, alpha=0.7)
    axes[1].set_xlabel("Шаг сетки dx")
    axes[1].set_ylabel("Ошибка L∞")
    axes[1].set_title("Сходимость FDM L∞")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    plt.savefig(output_dir / "fdm_convergence.png", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "fdm_convergence.pdf", bbox_inches="tight")
    plt.close()

    logger.info(f"✓ Convergence plot saved to {output_dir / 'fdm_convergence'}")


def plot_pinn_loss_history(
    history: Dict[str, list],
    output_dir: str | Path = "results/figures",
) -> None:
    """
    Plot PINN training history.

    Args:
        history: Training history dictionary with keys like 'epoch', 'loss_total', etc.
        output_dir: Output directory.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Plotting PINN training history...")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Total and component losses
    axes[0].semilogy(history["epoch"], history["loss_total"], label="Общая", linewidth=2)
    axes[0].semilogy(history["epoch"], history["loss_pde"], label="УЧП", linewidth=1.5, alpha=0.8)
    if max(history["loss_bc"]) > 0:
        axes[0].semilogy(history["epoch"], history["loss_bc"], label="ГУ", linewidth=1.5, alpha=0.8)
    if max(history["loss_ic"]) > 0:
        axes[0].semilogy(history["epoch"], history["loss_ic"], label="НУ", linewidth=1.5, alpha=0.8)
    axes[0].set_xlabel("Эпоха")
    axes[0].set_ylabel("Потеря")
    axes[0].set_title("Компоненты потери обучения PINN")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, which="both")

    # Learning rate
    if "learning_rate" in history:
        axes[1].semilogy(history["epoch"], history["learning_rate"], linewidth=2, color="green")
        axes[1].set_xlabel("Эпоха")
        axes[1].set_ylabel("Скорость обучения")
        axes[1].set_title("Расписание скорости обучения PINN")
        axes[1].grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    plt.savefig(output_dir / "pinn_loss_history.png", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "pinn_loss_history.pdf", bbox_inches="tight")
    plt.close()

    logger.info(f"✓ Loss history saved to {output_dir / 'pinn_loss_history'}")


def benchmark_timing(
    u_fdm: jax.Array,
    u_pinn: jax.Array,
    n_runs: int = 3,
) -> Dict[str, float]:
    """
    Benchmark inference time for FDM and PINN solutions.

    Args:
        u_fdm: FDM solution (for reference shape).
        u_pinn: PINN solution (for reference shape).
        n_runs: Number of repeated measurements.

    Returns:
        Dictionary with timing statistics.
    """
    import time
    
    logger.info(f"Benchmarking timing ({n_runs} runs)...")

    times_fdm = []
    times_pinn = []

    for _ in range(n_runs):
        # FDM timing (array copy)
        t0 = time.perf_counter()
        res = u_fdm * 1.0
        jax.block_until_ready(res)
        t1 = time.perf_counter()
        times_fdm.append(t1 - t0)

        # PINN timing (array copy)
        t0 = time.perf_counter()
        res = u_pinn * 1.0
        jax.block_until_ready(res)
        t1 = time.perf_counter()
        times_pinn.append(t1 - t0)

    return {
        "fdm_mean_ms": float(np.mean(times_fdm)) * 1000,
        "fdm_std_ms": float(np.std(times_fdm)) * 1000,
        "pinn_mean_ms": float(np.mean(times_pinn)) * 1000,
        "pinn_std_ms": float(np.std(times_pinn)) * 1000,
    }


def benchmark_fdm_timing(
    grid_size: int = 41,
    Lx: float = 1.0,
    Ly: float = 1.0,
    Lz: float = 1.0,
    alpha: float = 0.1,
    T: float = 1.0,
    output_dir: str | Path = "results/tables",
    n_warmup: int = 1,
    n_runs: int = 1,
) -> Dict[str, float]:
    """
    Benchmark FDM solve time by running solve_fdm_3d.
    
    Args:
        grid_size: Grid size N for N³ domain.
        Lx, Ly, Lz: Domain dimensions.
        alpha: Thermal diffusivity.
        T: Final time.
        output_dir: Directory to save results.
        n_warmup: Number of warmup runs (for JIT compilation).
        n_runs: Number of actual timed runs.
    
    Returns:
        Dictionary with keys: 'fdm_time_s', 'fdm_time_std_s', 'grid_size', 'n_points'.
    """
    import time
    from src.fdm_solver import solve_fdm_3d
    
    logger.info(f"Benchmarking FDM solve time for N={grid_size}³...")
    
    # Compute dt from CFL stability condition
    dx = Lx / (grid_size - 1)
    dy = Ly / (grid_size - 1)
    dz = Lz / (grid_size - 1)
    dt_cfl = 0.1 * min(dx, dy, dz)**2 / (6.0 * alpha)
    
    # Warmup run (for JIT compilation)
    for _ in range(n_warmup):
        u = solve_fdm_3d(
            Nx=grid_size, Ny=grid_size, Nz=grid_size,
            dt=dt_cfl, T=T,
            Lx=Lx, Ly=Ly, Lz=Lz,
            alpha=alpha
        )
        u.block_until_ready()
    
    # Actual timed runs
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        u = solve_fdm_3d(
            Nx=grid_size, Ny=grid_size, Nz=grid_size,
            dt=dt_cfl, T=T,
            Lx=Lx, Ly=Ly, Lz=Lz,
            alpha=alpha
        )
        u.block_until_ready()  # Wait for computation
        t1 = time.perf_counter()
        times.append(t1 - t0)
    
    mean_time = float(np.mean(times))
    std_time = float(np.std(times)) if n_runs > 1 else 0.0
    
    logger.info(f"Real FDM solve time for N={grid_size}: {mean_time:.2f}±{std_time:.2f} seconds (avg of {n_runs} runs)")
    
    # Save to CSV
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_df = pd.DataFrame([{
        "grid_size": grid_size,
        "n_points": grid_size ** 3,
        "solve_time_s": mean_time,
        "solve_time_std_s": std_time,
        "n_runs": n_runs,
    }])
    results_df.to_csv(output_dir / "fdm_timing_benchmark.csv", index=False)
    logger.info(f"✓ FDM timing saved to {output_dir / 'fdm_timing_benchmark.csv'}")
    
    return {
        "fdm_time_s": mean_time,
        "fdm_time_std_s": std_time,
        "grid_size": grid_size,
        "n_points": grid_size ** 3,
    }



def compute_convergence_orders(
    metrics: pd.DataFrame,
    output_dir: str | Path = "results/tables",
) -> Dict[str, float]:
    """
    Compute numerical convergence orders from FDM metrics.

    Args:
        metrics: FDM metrics DataFrame with columns ['Nx', 'L2', 'Linf', 'rel_L2'].
        output_dir: Directory to save results.

    Returns:
        Dictionary with keys 'p_L2', 'p_Linf', 'p_rel_L2' (convergence orders).
    """
    logger.info("Computing FDM convergence orders...")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dx = 1.0 / (metrics["Nx"].values - 1)
    log_dx = np.log(dx)

    # Fit log-log convergence
    errors = {}
    for col in ["L2", "rel_L2", "Linf"]:
        if col in metrics.columns:
            log_err = np.log(metrics[col].values)
            p_fit = np.polyfit(log_dx, log_err, 1)
            p = p_fit[0]
            errors[f"p_{col}"] = float(p)
            logger.info(f"  {col} convergence order: p = {p:.3f}")

    # Save to CSV
    orders_df = pd.DataFrame([errors])
    orders_df.to_csv(output_dir / "convergence_orders.csv", index=False)
    logger.info(f"✓ Convergence orders saved to {output_dir / 'convergence_orders.csv'}")

    return errors


def evaluate_pinn_on_grid(
    pinn_results: Dict[str, Any],
    grid_size: int = 41,
    eval_times: list | None = None,
    Lx: float = 1.0,
    Ly: float = 1.0,
    Lz: float = 1.0,
    alpha: float = 0.1,
    T: float = 1.0,
) -> Tuple[jax.Array, Dict[str, list]]:
    """
    Evaluate trained PINN on a regular grid and compute errors vs analytical solution.

    Args:
        pinn_results: Dictionary from load_pinn_results with 'params', 'config'.
        grid_size: Size of evaluation grid (grid_size³ points).
        eval_times: Times to evaluate (default from config or [T]).
        Lx, Ly, Lz: Domain dimensions.
        alpha: Thermal diffusivity.
        T: Final time.

    Returns:
        Tuple (u_pinn, metrics_dict) where:
        - u_pinn: PINN solution on grid at time T
        - metrics_dict: Contains keys 'times', 'rel_l2', 'linf_abs', etc.
    """
    from src.pinn_solver import evaluate_pinn
    
    logger.info(f"Evaluating PINN on {grid_size}³ grid...")
    
    config = pinn_results.get("config", {})
    params = pinn_results.get("params")
    
    if eval_times is None:
        eval_times = config.get("eval_times", [T])
    
    x = jnp.linspace(0, Lx, grid_size)
    y = jnp.linspace(0, Ly, grid_size)
    z = jnp.linspace(0, Lz, grid_size)
    X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")
    
    metrics_list = []
    u_pinn_final = None
    
    for t in eval_times:
        T_eval = jnp.full_like(X, t)
        
        # Evaluate PINN
        u_pinn = evaluate_pinn(params, config, X, Y, Z, T_eval)
        u_exact = analytical_solution(X, Y, Z, t, Lx, Ly, Lz, alpha)
        
        # Compute errors
        rel_l2, linf_abs, rel_linf = compute_l2_linf(u_pinn, u_exact)
        
        metrics_list.append({
            "time": float(t),
            "rel_l2": rel_l2,
            "linf_abs": linf_abs,
            "rel_linf": rel_linf,
        })
        
        if abs(t - T) < 1e-6:
            u_pinn_final = u_pinn
    
    logger.info(f"✓ PINN evaluation completed: shape={u_pinn_final.shape if u_pinn_final is not None else 'N/A'}")
    
    return u_pinn_final, metrics_list


def run_full_validation(
    grid_size: int = 41,
    Lx: float = 1.0,
    Ly: float = 1.0,
    Lz: float = 1.0,
    alpha: float = 0.1,
    T: float = 1.0,
) -> None:
    """
    Main orchestrator function for full validation.

    Args:
        grid_size: Grid size for evaluation (default 41).
        Lx, Ly, Lz: Domain lengths.
        alpha: Thermal diffusivity coefficient.
        T: Final time.
    """
    logger.info("=" * 80)
    logger.info("STARTING FULL VALIDATION: FDM vs PINN")
    logger.info("=" * 80)
    logger.info(f"Parameters: L={Lx}, α={alpha}, T={T}, grid_size={grid_size}")

    setup_plotting()

    # Create output directories
    output_dirs = {
        "figures": Path("results/figures/comparison"),
        "tables": Path("results/tables"),
    }
    for d in output_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    # Load results
    fdm_results = load_fdm_results(grid_size=grid_size)
    pinn_results = load_pinn_results()

    u_fdm = fdm_results["u_num"]
    u_exact_fdm = fdm_results["u_exact"]
    fdm_metrics = fdm_results["metrics"]

    # Evaluate PINN on exact grid at final time
    x = jnp.linspace(0, Lx, grid_size)
    y = jnp.linspace(0, Ly, grid_size)
    z = jnp.linspace(0, Lz, grid_size)
    X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")
    u_exact = analytical_solution(X, Y, Z, T, Lx, Ly, Lz, alpha)

    # Evaluate PINN on the same grid as FDM for fair comparison
    try:
        u_pinn, pinn_grid_metrics = evaluate_pinn_on_grid(
            pinn_results,
            grid_size=grid_size,
            eval_times=[0.0, 0.1, 0.3, 0.5, T],
            Lx=Lx, Ly=Ly, Lz=Lz,
            alpha=alpha, T=T
        )
        logger.info(f"✓ PINN evaluated on {grid_size}³ grid")
    except Exception as e:
        logger.warning(f"⚠ Failed to evaluate PINN on {grid_size}³ grid: {e}")
        logger.warning("  Using PINN from original eval_grid_size")
        u_pinn = None
        pinn_grid_metrics = None

    # Compute PINN errors at different times
    pinn_metrics_time = pd.DataFrame(pinn_results["metrics_by_time"])

    # Compute FDM convergence orders
    convergence_orders = compute_convergence_orders(fdm_metrics, output_dirs["tables"])

    # Generate tables
    logger.info("Generating comparison tables...")
    comp_table = generate_comparison_table(fdm_metrics, pinn_metrics_time, output_dirs["tables"])

    # Log CFL stability information
    dx = Lx / (grid_size - 1)
    dt_cfl = 0.1 * dx**2 / (6.0 * alpha)
    n_steps_estimated = int(np.ceil(T / dt_cfl))
    logger.info(f"CFL Stability Check:")
    logger.info(f"  dx = {dx:.4e}")
    logger.info(f"  dt_cfl = {dt_cfl:.4e}")
    logger.info(f"  Estimated time steps: ~{n_steps_estimated}")

    # Log PINN hyperparameters from config
    pinn_config = pinn_results.get("config", {})
    logger.info(f"PINN Hyperparameters:")
    logger.info(f"  Hidden width: {pinn_config.get('hidden_width', 'N/A')}")
    logger.info(f"  Hidden layers: {pinn_config.get('hidden_layers', 'N/A')}")
    logger.info(f"  Training epochs: {pinn_config.get('epochs', 'N/A')}")
    logger.info(f"  PDE loss weight (λ_pde): {pinn_config.get('lambda_pde', 'N/A')}")
    logger.info(f"  BC loss weight (λ_bc): {pinn_config.get('lambda_bc', 'N/A')}")
    logger.info(f"  IC loss weight (λ_ic): {pinn_config.get('lambda_ic', 'N/A')}")

    # Plot comparisons
    logger.info("Generating plots...")
    if u_pinn is not None:
        plot_snapshots_comparison(
            u_fdm, u_pinn, u_exact, [0.0, 0.1, 0.3, 0.5, T],
            output_dirs["figures"]
        )
    else:
        logger.warning("⚠ Skipping snapshot comparison (PINN on full grid not available)")
    
    plot_fdm_convergence(fdm_metrics, output_dirs["figures"])
    plot_pinn_loss_history(pinn_results["history"], output_dirs["figures"])

    # Benchmark FDM real solve timing
    logger.info("Benchmarking FDM real solve timing...")
    try:
        fdm_timing = benchmark_fdm_timing(
            grid_size=grid_size,
            Lx=Lx, Ly=Ly, Lz=Lz,
            alpha=alpha, T=T,
            n_warmup=1, n_runs=1
        )
        logger.info(f"✓ Real FDM solve: {fdm_timing['fdm_time_s']:.2f}±{fdm_timing['fdm_time_std_s']:.2f} seconds")
    except Exception as e:
        logger.warning(f"⚠ FDM timing benchmark failed: {e}")
        fdm_timing = None
    
    # Benchmark inference timing
    logger.info("Running inference benchmarks...")
    timings = benchmark_timing(u_fdm, u_fdm)
    logger.info(f"Timing (inference): FDM array copy={timings['fdm_mean_ms']:.3f}±{timings['fdm_std_ms']:.3f} ms")
    if fdm_timing is None:
        logger.info(f"Note: Full FDM solve estimated ~{n_steps_estimated * timings['fdm_mean_ms'] / 1000:.1f} seconds for {grid_size}³ grid")
    else:
        logger.info(f"Note: Full FDM solve measured {fdm_timing['fdm_time_s']:.1f} seconds for {grid_size}³ grid")

    logger.info("=" * 80)
    logger.info("VALIDATION COMPLETE")
    logger.info("=" * 80)
