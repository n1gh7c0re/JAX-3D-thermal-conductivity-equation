import jax
import jax.numpy as jnp
from jax import jit, lax

jax.config.update("jax_enable_x64", True)


def create_initial_condition(
    Nx: int, Ny: int, Nz: int,
    Lx: float = 1.0, Ly: float = 1.0, Lz: float = 1.0
) -> jnp.ndarray:
    """Начальное условие: u(x,y,z,0) = sin(πx/Lx) sin(πy/Ly) sin(πz/Lz)"""
    x = jnp.linspace(0.0, Lx, Nx)
    y = jnp.linspace(0.0, Ly, Ny)
    z = jnp.linspace(0.0, Lz, Nz)
    X, Y, Z = jnp.meshgrid(x, y, z, indexing='ij')
    return jnp.sin(jnp.pi * X / Lx) * jnp.sin(jnp.pi * Y / Ly) * jnp.sin(jnp.pi * Z / Lz)


@jit
def laplacian_3d(u: jnp.ndarray, dx: float, dy: float, dz: float) -> jnp.ndarray:
    d2uxx = (u[2:, 1:-1, 1:-1] - 2.0 * u[1:-1, 1:-1, 1:-1] + u[:-2, 1:-1, 1:-1]) / dx**2
    d2uyy = (u[1:-1, 2:, 1:-1] - 2.0 * u[1:-1, 1:-1, 1:-1] + u[1:-1, :-2, 1:-1]) / dy**2
    d2uzz = (u[1:-1, 1:-1, 2:] - 2.0 * u[1:-1, 1:-1, 1:-1] + u[1:-1, 1:-1, :-2]) / dz**2
    return d2uxx + d2uyy + d2uzz


@jit
def euler_step(u: jnp.ndarray, dt: float, alpha: float, dx: float, dy: float, dz: float) -> jnp.ndarray:
    """Один шаг явной схемы Эйлера"""
    lap = laplacian_3d(u, dx, dy, dz)
    u_next_interior = u[1:-1, 1:-1, 1:-1] + alpha * dt * lap
    return u.at[1:-1, 1:-1, 1:-1].set(u_next_interior)


def solve_fdm_3d(
    Nx: int, Ny: int, Nz: int,
    dt: float, T: float,
    alpha: float = 1.0,
    Lx: float = 1.0, Ly: float = 1.0, Lz: float = 1.0
) -> jnp.ndarray:
    """Основной JAX-солвер 3D уравнения теплопроводности"""
    dx = Lx / (Nx - 1)
    dy = Ly / (Ny - 1)
    dz = Lz / (Nz - 1)

    dt_max = min(dx, dy, dz)**2 / (6.0 * alpha)

    dt_used = dt
    if dt > dt_max:
        print(f"⚠️  ВНИМАНИЕ: dt = {dt:.2e} > dt_max = {dt_max:.2e}.")
        dt_used = dt_max * 0.95
        print(f"   dt автоматически скорректирован до {dt_used:.2e} для устойчивости.")

    n_steps = int(round(T / dt))
    if n_steps < 1:
        n_steps = 1

    u0 = create_initial_condition(Nx, Ny, Nz, Lx, Ly, Lz)

    def scan_body(carry, _):
        return euler_step(carry, dt, alpha, dx, dy, dz), None

    final_u, _ = lax.scan(scan_body, u0, None, length=n_steps)
    return final_u