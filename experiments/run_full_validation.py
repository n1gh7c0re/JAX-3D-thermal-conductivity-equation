"""
Main validation script for FDM vs PINN comparison on 3D heat equation.

This script orchestrates the full validation workflow:
- Loads FDM and PINN results
- Computes comprehensive error metrics
- Generates publication-quality plots and tables
- Produces a scientific validation report

Usage:
    python -m experiments.run_full_validation [options]

Examples:
    python -m experiments.run_full_validation --grid_size 41 --device cpu
    python -m experiments.run_full_validation --t_max 1.0
"""

from __future__ import annotations
import sys
import argparse
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import jax.numpy as jnp

from src.validation import run_full_validation
from src.analytical_solution import analytical_solution

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Full validation of FDM vs PINN for 3D heat equation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m experiments.run_full_validation --grid_size 41
  python -m experiments.run_full_validation --device gpu --t_max 1.0
        """,
    )

    parser.add_argument(
        "--grid_size",
        type=int,
        default=41,
        help="Grid size N for N³ domain (default 41)",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "gpu"],
        default="cpu",
        help="Device to use (default cpu)",
    )
    parser.add_argument(
        "--t_max",
        type=float,
        default=1.0,
        help="Final time for validation (default 1.0)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help="Thermal diffusivity coefficient (default 0.1)",
    )
    parser.add_argument(
        "--lx",
        type=float,
        default=1.0,
        help="Domain length in x (default 1.0)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Perform dry run (validate parameters without full compute)",
    )

    return parser.parse_args()


def validate_environment() -> None:
    """Check that all required result files exist."""
    logger.info("Validating environment...")

    required_dirs = [
        Path("results/fdm"),
        Path("results/pinn"),
        Path("src"),
        Path("experiments"),
    ]

    for d in required_dirs:
        if not d.exists():
            raise FileNotFoundError(f"Required directory not found: {d}")
        logger.info(f"✓ {d} exists")

    required_files = [
        Path("results/fdm/fdm_u_num_N41.npy"),
        Path("results/fdm/fdm_u_exact_N41.npy"),
        Path("results/fdm/metrics.csv"),
        Path("results/pinn/params.pkl"),
        Path("results/pinn/history.json"),
        Path("results/pinn/metrics_by_time.json"),
    ]

    for f in required_files:
        if not f.exists():
            logger.warning(f"⚠ Expected file not found: {f}")
            logger.warning("  Run experiments/run_fdm_validation.py and run_pinn_validation.py first")
        else:
            logger.info(f"✓ {f} exists")


def perform_dry_run(args: argparse.Namespace) -> None:
    """Perform a dry run: validate parameters and check analytical solution."""
    logger.info("Running dry run mode...")

    logger.info(f"Parameters: α={args.alpha}, L={args.lx}, T={args.t_max}, grid_size={args.grid_size}")

    # Test analytical solution evaluation
    try:
        x = jnp.linspace(0, args.lx, args.grid_size)
        X, Y, Z = jnp.meshgrid(x, x, x, indexing="ij")
        u = analytical_solution(X, Y, Z, args.t_max, args.lx, args.lx, args.lx, args.alpha)
        logger.info(f"✓ Analytical solution evaluated: shape={u.shape}, range=[{float(jnp.min(u)):.3e}, {float(jnp.max(u)):.3e}]")
    except Exception as e:
        logger.error(f"✗ Analytical solution evaluation failed: {e}", exc_info=True)
        raise

    # Check CFL stability
    dx = args.lx / (args.grid_size - 1)
    dt_cfl = 0.1 * dx**2 / (6.0 * args.alpha)
    n_steps = int(np.ceil(args.t_max / dt_cfl))
    logger.info(f"✓ CFL parameters: dx={dx:.4e}, dt_cfl={dt_cfl:.4e}, n_steps={n_steps}")

    logger.info("Dry run completed successfully. Ready for full validation.")


def main() -> None:
    """Main function."""
    args = parse_arguments()

    logger.info("=" * 80)
    logger.info("FULL VALIDATION WORKFLOW: FDM vs PINN (3D Heat Equation)")
    logger.info("=" * 80)

    # Validate environment
    try:
        validate_environment()
    except FileNotFoundError as e:
        logger.error(f"Environment validation failed: {e}")
        logger.error("Please run experiments/run_fdm_validation.py and experiments/run_pinn_validation.py first")
        sys.exit(1)

    if args.dry_run:
        perform_dry_run(args)
        logger.info("Dry run completed. Exiting.")
        sys.exit(0)

    # Run full validation
    try:
        run_full_validation(
            grid_size=args.grid_size,
            Lx=args.lx,
            Ly=args.lx,
            Lz=args.lx,
            alpha=args.alpha,
            T=args.t_max,
        )
        logger.info("Full validation completed successfully!")
    except Exception as e:
        logger.error(f"Validation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
