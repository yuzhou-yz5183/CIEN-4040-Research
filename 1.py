import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re


def _validate_model_inputs(v, A, B, gamma):
    v = np.asarray(v, dtype=float)
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    gamma = np.asarray(gamma, dtype=float)

    if v.ndim != 1:
        raise ValueError("v must be a one-dimensional array.")

    n_classes = v.size
    expected_shape = (n_classes, n_classes)

    if A.shape != expected_shape or B.shape != expected_shape or gamma.shape != expected_shape:
        raise ValueError("A, B, and gamma must all have shape (N, N).")

    if np.any(np.diff(v) <= 0):
        raise ValueError("The desired speeds in v must satisfy v1 < v2 < ... < vN.")

    return v, A, B, gamma


def _default_initial_condition(x, n_classes):
    bump = np.exp(-((x - 0.4) ** 2) / (2 * 0.08 ** 2))
    return np.repeat((bump / n_classes)[None, :], n_classes, axis=0)


def _compute_speed(f, v, P, eps, look_ahead_speed=True):
    rho = np.sum(f, axis=0)
    weighted_speed = np.sum(f * v[:, None], axis=0)

    # The model uses a downstream-looking speed law rho_{j+1}^k by default.
    # This keeps the total density bounded much better than the purely local
    # alternative, while the flux itself remains standard upwind.
    rho_for_speed = np.roll(rho, -1) if look_ahead_speed else rho
    weighted_speed_for_limiter = np.roll(weighted_speed, -1) if look_ahead_speed else weighted_speed
    weighted_speed_for_limiter = np.maximum(weighted_speed_for_limiter, eps)
    limiter = rho_for_speed * v[-1] * (1.0 - rho_for_speed / P) / weighted_speed_for_limiter
    limiter = np.clip(limiter, 0.0, 1.0)

    u = v[:, None] * limiter[None, :]
    return u, rho


def _compute_transition_rates(rho, P, A, B, gamma):
    n_classes = gamma.shape[0]
    n_cells = rho.size
    eta_up = np.zeros((n_classes, n_classes, n_cells))
    eta_down = np.zeros((n_classes, n_classes, n_cells))

    free_space = np.maximum(P - rho, 0.0)
    occupied = np.maximum(rho, 0.0)

    for n in range(n_classes):
        if n < n_classes - 1:
            eta_up[n, :, :] = free_space[None, :] * gamma[n, :, None] + A[n, :, None]
        if n > 0:
            eta_down[n, :, :] = occupied[None, :] * gamma[n, :, None] + B[n, :, None]

    return eta_up, eta_down


def _compute_reaction(f, eta_up, eta_down):
    n_classes, n_cells = f.shape
    reaction = np.zeros_like(f)

    for n in range(n_classes):
        term = np.zeros(n_cells)

        if n > 0:
            gain_from_lower = np.sum(f * eta_up[n - 1], axis=0)
            term += f[n - 1] * gain_from_lower

        loss_rate = np.sum(f * (eta_up[n] + eta_down[n]), axis=0)
        term -= f[n] * loss_rate

        if n < n_classes - 1:
            gain_from_upper = np.sum(f * eta_down[n + 1], axis=0)
            term += f[n + 1] * gain_from_upper

        reaction[n] = term

    return reaction


def simulate_multiclass(
    J=200,
    L=1.0,
    T=0.8,
    P=1.0,
    v=(0.5, 1.0),
    A=None,
    B=None,
    gamma=None,
    dt=None,
    initial_condition=None,
    eps=1e-12,
    enforce_density_cap=False,
    look_ahead_speed=True,
):
    v = np.asarray(v, dtype=float)
    n_classes = v.size

    if A is None:
        A = np.zeros((n_classes, n_classes))
    if B is None:
        B = np.zeros((n_classes, n_classes))
    if gamma is None:
        gamma = np.zeros((n_classes, n_classes))

    v, A, B, gamma = _validate_model_inputs(v, A, B, gamma)

    dx = L / J
    vmax = v[-1]

    if dt is None:
        dt = 0.4 * dx / max(vmax, eps)

    n_steps = max(1, int(np.ceil(T / dt)))
    dt = T / n_steps

    x = np.linspace(0.0, L, J, endpoint=False)
    t_grid = np.linspace(0.0, T, n_steps + 1)

    if initial_condition is None:
        f = _default_initial_condition(x, n_classes)
    elif callable(initial_condition):
        f = np.asarray(initial_condition(x, v), dtype=float)
    else:
        f = np.asarray(initial_condition, dtype=float)

    if f.shape != (n_classes, J):
        raise ValueError("initial_condition must produce an array with shape (N, J).")

    f = np.maximum(f, 0.0)

    if enforce_density_cap:
        rho0 = np.sum(f, axis=0)
        mask = rho0 > P
        if np.any(mask):
            f[:, mask] *= (P / rho0[mask])[None, :]

    f_history = np.zeros((n_steps + 1, n_classes, J))
    rho_history = np.zeros((n_steps + 1, J))
    f_history[0] = f
    rho_history[0] = np.sum(f, axis=0)

    for k in range(n_steps):
        u, rho = _compute_speed(f, v, P, eps, look_ahead_speed=look_ahead_speed)
        flux = u * f
        # Standard first-order upwind flux for nonnegative speeds.
        advection = -(dt / dx) * (flux - np.roll(flux, 1, axis=1))

        eta_up, eta_down = _compute_transition_rates(rho, P, A, B, gamma)
        reaction = _compute_reaction(f, eta_up, eta_down)

        f = f + advection + dt * reaction
        f = np.maximum(f, 0.0)

        if enforce_density_cap:
            rho_new = np.sum(f, axis=0)
            mask = rho_new > P
            if np.any(mask):
                # This cap is non-conservative: it removes mass whenever rho > P.
                f[:, mask] *= (P / rho_new[mask])[None, :]

        f_history[k + 1] = f
        rho_history[k + 1] = np.sum(f, axis=0)

    return x, t_grid, f_history, rho_history


def simulate_two_class(
    J=200,
    L=1.0,
    T=0.8,
    P=1.0,
    v=(0.5, 1.0),
    A12=0.1,
    B21=0.1,
    gamma=((0.1, 0.2), (0.3, 0.4)),
    dt=None,
    initial_condition=None,
    eps=1e-12,
    enforce_density_cap=False,
    look_ahead_speed=True,
):
    A = np.array([[0.0, A12], [0.0, 0.0]], dtype=float)
    B = np.array([[0.0, 0.0], [B21, 0.0]], dtype=float)
    gamma = np.asarray(gamma, dtype=float)

    x, t_grid, f_history, rho_history = simulate_multiclass(
        J=J,
        L=L,
        T=T,
        P=P,
        v=v,
        A=A,
        B=B,
        gamma=gamma,
        dt=dt,
        initial_condition=initial_condition,
        eps=eps,
        enforce_density_cap=enforce_density_cap,
        look_ahead_speed=look_ahead_speed,
    )

    return x, t_grid, f_history, rho_history


def _figure_path_from_title(title):
    figure_dir = Path(__file__).resolve().parent / "figure"
    figure_dir.mkdir(exist_ok=True)
    safe_title = re.sub(r"[^A-Za-z0-9._-]+", "_", title).strip("._")
    safe_title = safe_title or "figure"
    return figure_dir / f"{safe_title}.png"


def _clip_density_for_plot(density_map, vmin=0.0, vmax=1.0):
    density_map = np.asarray(density_map, dtype=float)
    return np.clip(density_map, vmin, vmax)


def plot_heatmap(
    x,
    t_grid,
    rho_map,
    title="Total density heat map",
    save_path=None,
    show=True,
    vmin=0.0,
    vmax=1.0,
):
    rho_map = _clip_density_for_plot(rho_map, vmin=vmin, vmax=vmax)

    plt.figure(figsize=(10, 6))
    plt.imshow(
        rho_map,
        aspect="auto",
        origin="lower",
        extent=[x[0], x[-1], t_grid[0], t_grid[-1]],
        cmap="RdYlBu_r",
        vmin=vmin,
        vmax=vmax,
    )
    plt.colorbar(label="Density")
    plt.xlabel("Space x")
    plt.ylabel("Time t")
    plt.title(title)

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def plot_class_heatmap(
    x,
    t_grid,
    f_history,
    class_index,
    title=None,
    colorbar_label=None,
    save_path=None,
    show=True,
    vmin=0.0,
    vmax=1.0,
):
    class_map = _clip_density_for_plot(f_history[:, class_index, :], vmin=vmin, vmax=vmax)

    plt.figure(figsize=(10, 6))
    plt.imshow(
        class_map,
        aspect="auto",
        origin="lower",
        extent=[x[0], x[-1], t_grid[0], t_grid[-1]],
        cmap="RdYlBu_r",
        vmin=vmin,
        vmax=vmax,
    )
    plt.colorbar(label=colorbar_label or f"f{class_index + 1}")
    plt.xlabel("Space x")
    plt.ylabel("Time t")
    plt.title(title or f"Density heat map for class {class_index + 1}")

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def plot_slow_car_heatmap(
    x,
    t_grid,
    f_history,
    title="Slow Car Density Heat Map",
    save_path=None,
    show=True,
    vmin=0.0,
    vmax=1.0,
):
    plot_class_heatmap(
        x,
        t_grid,
        f_history,
        class_index=0,
        title=title,
        colorbar_label="Slow car density",
        save_path=save_path,
        show=show,
        vmin=vmin,
        vmax=vmax,
    )


def plot_fast_car_heatmap(
    x,
    t_grid,
    f_history,
    title="Fast Car Density Heat Map",
    save_path=None,
    show=True,
    vmin=0.0,
    vmax=1.0,
):
    plot_class_heatmap(
        x,
        t_grid,
        f_history,
        class_index=1,
        title=title,
        colorbar_label="Fast car density",
        save_path=save_path,
        show=show,
        vmin=vmin,
        vmax=vmax,
    )


def plot_final_profiles(x, rho_map, vmin=0.0, vmax=1.0, save_path=None, show=True):
    rho_map = _clip_density_for_plot(rho_map, vmin=vmin, vmax=vmax)
    title = "Initial vs Final total density"
    plt.figure(figsize=(10, 5))
    plt.plot(x, rho_map[0], label="Initial density")
    plt.plot(x, rho_map[-1], label="Final density")
    plt.xlabel("Space x")
    plt.ylabel("Density")
    plt.ylim(vmin, vmax)
    plt.title(title)
    plt.legend()

    if save_path is None:
        save_path = _figure_path_from_title(title)

    plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    # Tune the parameters here to explore different scenarios. The current defaults are just a starting point.
    A12=0.1
    B21=0.1
    gamma=((0.1, 0.2), (0.3, 0.4))
    x, t_grid, f_history, rho_map = simulate_two_class(J=200, T=1.0, A12=A12, B21=B21,  gamma=gamma)

    total_title = "Total density heat map with parameters ({}, {}, {})".format(A12, B21, gamma)
    slow_title = "Slow Car Density Heat Map with parameters ({}, {}, {})".format(A12, B21, gamma)
    fast_title = "Fast Car Density Heat Map with parameters ({}, {}, {})".format(A12, B21, gamma)

    plot_heatmap(
        x,
        t_grid,
        rho_map,
        title=total_title,
        save_path=_figure_path_from_title("Total with {},{},{}".format(A12, B21, gamma)),
    )
    plot_slow_car_heatmap(
        x,
        t_grid,
        f_history,
        title=slow_title,
        save_path=_figure_path_from_title("Slow with {},{},{}".format(A12, B21, gamma)),
    )
    plot_fast_car_heatmap(
        x,
        t_grid,
        f_history,
        title=fast_title,
        save_path=_figure_path_from_title("Fast with {},{},{}".format(A12, B21, gamma)),
    )
    plot_final_profiles(x, rho_map)
