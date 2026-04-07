"""
Analytical solution for 3D heat equation with Dirichlet boundary conditions.

The heat equation:
    u_t = α(u_xx + u_yy + u_zz)

Domain: Ω = [0, Lx] × [0, Ly] × [0, Lz]
Boundary conditions: u = 0 on ∂Ω
Initial condition: u(x,y,z,0) = sin(πx/Lx) sin(πy/Ly) sin(πz/Lz)

Analytical solution (separation of variables):
    u(x,y,z,t) = exp(-α λ t) sin(πx/Lx) sin(πy/Ly) sin(πz/Lz)
where λ = π²(1/Lx² + 1/Ly² + 1/Lz²)
"""

from __future__ import annotations
import jax
import jax.numpy as jnp
from typing import Union

__all__ = ["analytical_solution", "decay_rate"]


def decay_rate(
    Lx: float = 1.0,
    Ly: float = 1.0,
    Lz: float = 1.0
) -> float:
    """
    Compute the spectral decay rate λ = π²(1/Lx² + 1/Ly² + 1/Lz²).

    Args:
        Lx: Domain length in x direction.
        Ly: Domain length in y direction.
        Lz: Domain length in z direction.

    Returns:
        Decay rate λ.
    """
    return jnp.pi**2 * (1.0 / Lx**2 + 1.0 / Ly**2 + 1.0 / Lz**2)


def analytical_solution(
    x: jax.Array,
    y: jax.Array,
    z: jax.Array,
    t: Union[float, jax.Array],
    Lx: float = 1.0,
    Ly: float = 1.0,
    Lz: float = 1.0,
    alpha: float = 0.1
) -> jax.Array:
    """
    Compute the analytical solution of the 3D heat equation.

    Supports both meshgrid and flattened coordinate arrays. Fully JIT-compatible.

    Args:
        x: Spatial coordinate(s) in x direction. Shape can be (...,) or scalar.
        y: Spatial coordinate(s) in y direction. Same shape as x.
        z: Spatial coordinate(s) in z direction. Same shape as x.
        t: Time value(s). Scalar or array broadcastable with x, y, z.
        Lx: Domain length in x direction (default 1.0).
        Ly: Domain length in y direction (default 1.0).
        Lz: Domain length in z direction (default 1.0).
        alpha: Thermal diffusivity coefficient (default 0.1).

    Returns:
        Solution u(x, y, z, t) with same shape as (broadcasted) x, y, z.

    Examples:
        # Single point
        u = analytical_solution(0.5, 0.5, 0.5, 0.1)

        # Meshgrid (3D)
        X, Y, Z = jnp.meshgrid(x_array, y_array, z_array, indexing='ij')
        U = analytical_solution(X, Y, Z, 0.1)

        # Time evolution at single point
        times = jnp.linspace(0, 1, 100)
        u_t = analytical_solution(0.5, 0.5, 0.5, times)
    """
    # Compute decay factor
    lam = decay_rate(Lx, Ly, Lz)
    temporal_decay = jnp.exp(-alpha * lam * t)

    # Spatial sinusoidal modes
    spatial_part = (
        jnp.sin(jnp.pi * x / Lx)
        * jnp.sin(jnp.pi * y / Ly)
        * jnp.sin(jnp.pi * z / Lz)
    )

    return temporal_decay * spatial_part
