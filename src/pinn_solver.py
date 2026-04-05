from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json
import pickle
from typing import Dict, Tuple
from functools import partial

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


Array = jnp.ndarray


def tree_zeros_like(tree):
    return jax.tree_util.tree_map(jnp.zeros_like, tree)


class AdamOptimizer:

    def __init__(self, learning_rate: float, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

    def init(self, params):
        zeros = tree_zeros_like(params)
        return {"step": jnp.array(0, dtype=jnp.int32), "m": zeros, "v": zeros}

    def update(self, params, grads, state):
        step = state["step"] + jnp.array(1, dtype=jnp.int32)
        m = jax.tree_util.tree_map(
            lambda m_old, g: self.beta1 * m_old + (1.0 - self.beta1) * g,
            state["m"], grads
        )
        v = jax.tree_util.tree_map(
            lambda v_old, g: self.beta2 * v_old + (1.0 - self.beta2) * (g ** 2),
            state["v"], grads
        )

        m_hat = jax.tree_util.tree_map(lambda x: x / (1.0 - self.beta1 ** step), m)
        v_hat = jax.tree_util.tree_map(lambda x: x / (1.0 - self.beta2 ** step), v)

        new_params = jax.tree_util.tree_map(
            lambda p, mh, vh: p - self.learning_rate * mh / (jnp.sqrt(vh) + self.eps),
            params, m_hat, v_hat
        )
        return new_params, {"step": step, "m": m, "v": v}


@dataclass(frozen=True)
class PINNConfig:
    alpha: float = 1.0
    Lx: float = 1.0
    Ly: float = 1.0
    Lz: float = 1.0
    T: float = 0.1

    hidden_width: int = 64
    hidden_layers: int = 4

    n_residual: int = 10000
    n_boundary: int = 3000
    n_initial: int = 3000

    learning_rate: float = 1e-3
    epochs: int = 5000
    print_every: int = 250

    lambda_pde: float = 1.0
    lambda_bc: float = 10.0
    lambda_ic: float = 10.0

    eval_grid_size: int = 21
    eval_times: Tuple[float, ...] = (0.0, 0.02, 0.05, 0.1)
    seed: int = 42


def exact_solution(points: Array, alpha: float = 1.0,
                   Lx: float = 1.0, Ly: float = 1.0, Lz: float = 1.0) -> Array:
    """
    Exact solution for:
        u_t = alpha * (u_xx + u_yy + u_zz)
    with homogeneous Dirichlet BC and initial condition
        sin(pi x/Lx) sin(pi y/Ly) sin(pi z/Lz)
    """
    x, y, z, t = points[:, 0], points[:, 1], points[:, 2], points[:, 3]
    decay = jnp.pi ** 2 * (1.0 / Lx ** 2 + 1.0 / Ly ** 2 + 1.0 / Lz ** 2)
    u = jnp.exp(-alpha * decay * t)
    u = u * jnp.sin(jnp.pi * x / Lx) * jnp.sin(jnp.pi * y / Ly) * jnp.sin(jnp.pi * z / Lz)
    return u[:, None]


def _glorot_std(in_dim: int, out_dim: int) -> float:
    return jnp.sqrt(2.0 / (in_dim + out_dim))


def init_mlp_params(layer_sizes, key) -> list[dict[str, Array]]:
    keys = jax.random.split(key, len(layer_sizes) - 1)
    params = []
    for k, (in_dim, out_dim) in zip(keys, zip(layer_sizes[:-1], layer_sizes[1:])):
        W = _glorot_std(in_dim, out_dim) * jax.random.normal(k, (in_dim, out_dim))
        b = jnp.zeros((out_dim,))
        params.append({"W": W, "b": b})
    return params


def forward_mlp(params: list[dict[str, Array]], x: Array) -> Array:
    h = x
    for layer in params[:-1]:
        h = jnp.tanh(h @ layer["W"] + layer["b"])
    out = h @ params[-1]["W"] + params[-1]["b"]
    return out


def normalize_inputs(points: Array, cfg: PINNConfig) -> Array:
    """Map [0,Lx]x[0,Ly]x[0,Lz]x[0,T] to [-1,1]^4."""
    x = 2.0 * points[:, 0:1] / cfg.Lx - 1.0
    y = 2.0 * points[:, 1:2] / cfg.Ly - 1.0
    z = 2.0 * points[:, 2:3] / cfg.Lz - 1.0
    t = 2.0 * points[:, 3:4] / cfg.T - 1.0 if cfg.T > 0 else points[:, 3:4]
    return jnp.concatenate([x, y, z, t], axis=1)


def predict(params: list[dict[str, Array]], points: Array, cfg: PINNConfig) -> Array:
    return forward_mlp(params, normalize_inputs(points, cfg))


def _scalar_predict(params: list[dict[str, Array]], point: Array, cfg: PINNConfig) -> Array:
    point_2d = point[None, :]
    return predict(params, point_2d, cfg).squeeze()


def pde_residual_single(params: list[dict[str, Array]], point: Array, cfg: PINNConfig) -> Array:
    """Residual r = u_t - alpha (u_xx + u_yy + u_zz) at one point."""
    grad_u = jax.grad(_scalar_predict, argnums=1)(params, point, cfg)
    u_t = grad_u[3]

    def du_dx(_point):
        return jax.grad(_scalar_predict, argnums=1)(params, _point, cfg)[0]

    def du_dy(_point):
        return jax.grad(_scalar_predict, argnums=1)(params, _point, cfg)[1]

    def du_dz(_point):
        return jax.grad(_scalar_predict, argnums=1)(params, _point, cfg)[2]

    u_xx = jax.grad(du_dx)(point)[0]
    u_yy = jax.grad(du_dy)(point)[1]
    u_zz = jax.grad(du_dz)(point)[2]
    return u_t - cfg.alpha * (u_xx + u_yy + u_zz)


pde_residual_batch = jax.jit(jax.vmap(pde_residual_single, in_axes=(None, 0, None)), static_argnums=(2,))


def sample_residual_points(key, n: int, cfg: PINNConfig) -> Array:
    key_x, key_y, key_z, key_t = jax.random.split(key, 4)
    x = jax.random.uniform(key_x, (n, 1), minval=0.0, maxval=cfg.Lx)
    y = jax.random.uniform(key_y, (n, 1), minval=0.0, maxval=cfg.Ly)
    z = jax.random.uniform(key_z, (n, 1), minval=0.0, maxval=cfg.Lz)
    t = jax.random.uniform(key_t, (n, 1), minval=0.0, maxval=cfg.T)
    return jnp.concatenate([x, y, z, t], axis=1)


def sample_initial_points(key, n: int, cfg: PINNConfig) -> tuple[Array, Array]:
    key_x, key_y, key_z = jax.random.split(key, 3)
    x = jax.random.uniform(key_x, (n, 1), minval=0.0, maxval=cfg.Lx)
    y = jax.random.uniform(key_y, (n, 1), minval=0.0, maxval=cfg.Ly)
    z = jax.random.uniform(key_z, (n, 1), minval=0.0, maxval=cfg.Lz)
    t = jnp.zeros((n, 1))
    pts = jnp.concatenate([x, y, z, t], axis=1)
    vals = exact_solution(pts, alpha=cfg.alpha, Lx=cfg.Lx, Ly=cfg.Ly, Lz=cfg.Lz)
    return pts, vals


def sample_boundary_points(key, n: int, cfg: PINNConfig) -> tuple[Array, Array]:
    if n < 6:
        raise ValueError("n_boundary must be at least 6")
    per_face = n // 6
    rem = n - 6 * per_face
    counts = [per_face] * 6
    for i in range(rem):
        counts[i] += 1

    face_keys = jax.random.split(key, 24)
    points = []
    idx = 0
    for face_id, count in enumerate(counts):
        if count == 0:
            continue
        k1, k2, k3, k4 = face_keys[idx:idx + 4]
        idx += 4
        if face_id in (0, 1):  # x = 0 or Lx
            x = jnp.zeros((count, 1)) if face_id == 0 else jnp.full((count, 1), cfg.Lx)
            y = jax.random.uniform(k1, (count, 1), minval=0.0, maxval=cfg.Ly)
            z = jax.random.uniform(k2, (count, 1), minval=0.0, maxval=cfg.Lz)
            t = jax.random.uniform(k3, (count, 1), minval=0.0, maxval=cfg.T)
        elif face_id in (2, 3):  # y = 0 or Ly
            x = jax.random.uniform(k1, (count, 1), minval=0.0, maxval=cfg.Lx)
            y = jnp.zeros((count, 1)) if face_id == 2 else jnp.full((count, 1), cfg.Ly)
            z = jax.random.uniform(k2, (count, 1), minval=0.0, maxval=cfg.Lz)
            t = jax.random.uniform(k3, (count, 1), minval=0.0, maxval=cfg.T)
        else:  # z = 0 or Lz
            x = jax.random.uniform(k1, (count, 1), minval=0.0, maxval=cfg.Lx)
            y = jax.random.uniform(k2, (count, 1), minval=0.0, maxval=cfg.Ly)
            z = jnp.zeros((count, 1)) if face_id == 4 else jnp.full((count, 1), cfg.Lz)
            t = jax.random.uniform(k3, (count, 1), minval=0.0, maxval=cfg.T)
        points.append(jnp.concatenate([x, y, z, t], axis=1))

    pts = jnp.concatenate(points, axis=0)
    vals = jnp.zeros((pts.shape[0], 1))
    return pts, vals


def make_training_batch(key, cfg: PINNConfig) -> Dict[str, Array]:
    kr, kb, ki = jax.random.split(key, 3)
    x_r = sample_residual_points(kr, cfg.n_residual, cfg)
    x_b, y_b = sample_boundary_points(kb, cfg.n_boundary, cfg)
    x_i, y_i = sample_initial_points(ki, cfg.n_initial, cfg)
    return {"x_r": x_r, "x_b": x_b, "y_b": y_b, "x_i": x_i, "y_i": y_i}


def loss_terms(params, batch: Dict[str, Array], cfg: PINNConfig):
    r = pde_residual_batch(params, batch["x_r"], cfg)
    loss_pde = jnp.mean(r ** 2)

    pred_b = predict(params, batch["x_b"], cfg)
    loss_bc = jnp.mean((pred_b - batch["y_b"]) ** 2)

    pred_i = predict(params, batch["x_i"], cfg)
    loss_ic = jnp.mean((pred_i - batch["y_i"]) ** 2)

    total = cfg.lambda_pde * loss_pde + cfg.lambda_bc * loss_bc + cfg.lambda_ic * loss_ic
    return total, {"loss_pde": loss_pde, "loss_bc": loss_bc, "loss_ic": loss_ic}


@partial(jax.jit, static_argnums=(2,))
def _loss_total_only(params, batch, cfg):
    total, _ = loss_terms(params, batch, cfg)
    return total


def create_train_state(cfg: PINNConfig):
    layer_sizes = [4] + [cfg.hidden_width] * cfg.hidden_layers + [1]
    key = jax.random.PRNGKey(cfg.seed)
    params = init_mlp_params(layer_sizes, key)
    optimizer = AdamOptimizer(cfg.learning_rate)
    opt_state = optimizer.init(params)
    return params, optimizer, opt_state


@partial(jax.jit, static_argnums=(3, 4))
def train_step(params, opt_state, batch, optimizer, cfg):
    def loss_fn(p):
        total, aux = loss_terms(p, batch, cfg)
        return total, aux

    (loss_value, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    params, opt_state = optimizer.update(params, grads, opt_state)

    metrics = {
        "loss_total": loss_value,
        "loss_pde": aux["loss_pde"],
        "loss_bc": aux["loss_bc"],
        "loss_ic": aux["loss_ic"],
    }
    return params, opt_state, metrics


def build_eval_grid(cfg: PINNConfig, time_value: float, n: int | None = None) -> Array:
    n = n or cfg.eval_grid_size
    x = jnp.linspace(0.0, cfg.Lx, n)
    y = jnp.linspace(0.0, cfg.Ly, n)
    z = jnp.linspace(0.0, cfg.Lz, n)
    X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")
    t = jnp.full_like(X, time_value)
    pts = jnp.stack([X, Y, Z, t], axis=-1).reshape(-1, 4)
    return pts


def compute_error_metrics(params, cfg: PINNConfig, time_value: float, n: int | None = None) -> Dict[str, float]:
    pts = build_eval_grid(cfg, time_value, n=n)
    u_pred = predict(params, pts, cfg)
    u_ex = exact_solution(pts, alpha=cfg.alpha, Lx=cfg.Lx, Ly=cfg.Ly, Lz=cfg.Lz)

    diff = u_pred - u_ex
    l2 = jnp.sqrt(jnp.mean(diff ** 2))
    linf = jnp.max(jnp.abs(diff))
    rel_l2 = l2 / (jnp.sqrt(jnp.mean(u_ex ** 2)) + 1e-12)

    return {
        "time": float(time_value),
        "l2": float(l2),
        "linf": float(linf),
        "rel_l2": float(rel_l2),
    }


def train_pinn(cfg: PINNConfig | None = None):
    cfg = cfg or PINNConfig()
    params, optimizer, opt_state = create_train_state(cfg)
    key = jax.random.PRNGKey(cfg.seed + 123)

    history = {
        "epoch": [],
        "loss_total": [],
        "loss_pde": [],
        "loss_bc": [],
        "loss_ic": [],
    }

    for epoch in range(1, cfg.epochs + 1):
        key, subkey = jax.random.split(key)
        batch = make_training_batch(subkey, cfg)
        params, opt_state, metrics = train_step(params, opt_state, batch, optimizer, cfg)

        history["epoch"].append(epoch)
        history["loss_total"].append(float(metrics["loss_total"]))
        history["loss_pde"].append(float(metrics["loss_pde"]))
        history["loss_bc"].append(float(metrics["loss_bc"]))
        history["loss_ic"].append(float(metrics["loss_ic"]))

        if epoch % cfg.print_every == 0 or epoch == 1 or epoch == cfg.epochs:
            print(
                f"[{epoch:5d}/{cfg.epochs}] "
                f"total={metrics['loss_total']:.3e}, "
                f"pde={metrics['loss_pde']:.3e}, "
                f"bc={metrics['loss_bc']:.3e}, "
                f"ic={metrics['loss_ic']:.3e}"
            )

    metrics_by_time = [compute_error_metrics(params, cfg, t) for t in cfg.eval_times]
    return params, history, metrics_by_time, cfg


def save_training_outputs(output_dir: str | Path, params, history, metrics_by_time, cfg: PINNConfig):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2, ensure_ascii=False)

    with open(output_dir / "metrics_by_time.json", "w", encoding="utf-8") as f:
        json.dump(metrics_by_time, f, indent=2, ensure_ascii=False)

    with open(output_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    with open(output_dir / "params.pkl", "wb") as f:
        pickle.dump(params, f)


def load_params(path: str | Path):
    with open(path, "rb") as f:
        return pickle.load(f)
