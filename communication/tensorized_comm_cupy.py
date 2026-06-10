"""CuPy backend for selected tensorized communication kernels."""

from __future__ import annotations

import numpy as np

try:
    import cupy as cp

    HAS_CUPY = True
except Exception:  # pragma: no cover
    cp = None
    HAS_CUPY = False


def _grouped_min_links() -> int:
    import os

    try:
        return int(os.environ.get("TENSORIZED_COMM_CUPY_GROUPED_MIN_LINKS", "512"))
    except Exception:
        return 512


def _compute_rates_cupy(
    actions_power: np.ndarray,
    destinations: np.ndarray,
    v2v_channels_with_fastfading: np.ndarray,
    v2i_channels_with_fastfading: np.ndarray,
    v2i_channels_abs: np.ndarray,
    *,
    power_db_list: list[float] | np.ndarray,
    v2i_power_db: float,
    veh_ant_gain: float,
    bs_ant_gain: float,
    bs_noise_figure: float,
    veh_noise_figure: float,
    sig2: float,
    n_rb: int,
    n_vehicles: int,
    active_links: np.ndarray | None = None,
    include_v2i_to_v2v: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not HAS_CUPY:
        raise RuntimeError("CuPy backend requested but cupy is unavailable")

    n_vehicles = int(n_vehicles)
    n_rb = int(n_rb)
    actions = cp.asarray(actions_power[:, :, 0], dtype=cp.int32)
    powers = cp.asarray(actions_power[:, :, 1], dtype=cp.int32)
    destinations = cp.asarray(destinations[:, :3], dtype=cp.int32)
    v2v_ch = cp.asarray(v2v_channels_with_fastfading, dtype=cp.float64)
    v2i_ch = cp.asarray(v2i_channels_with_fastfading, dtype=cp.float64)
    v2i_abs = cp.asarray(v2i_channels_abs, dtype=cp.float64)
    power_db = cp.asarray(power_db_list, dtype=cp.float64)

    veh_idx = cp.repeat(cp.arange(n_vehicles, dtype=cp.int32), 3)
    rb_flat = actions.reshape(-1)
    power_flat = powers.reshape(-1)
    dest_flat = destinations.reshape(-1)
    valid = (rb_flat >= 0) & (rb_flat < n_rb) & (dest_flat >= 0) & (dest_flat < n_vehicles)
    if active_links is not None:
        valid = valid & cp.asarray(active_links, dtype=cp.bool_).reshape(-1)
    links = cp.nonzero(valid)[0]

    v2i_interference = cp.zeros((n_rb,), dtype=cp.float64)
    v2v_signal = cp.zeros((n_vehicles * 3,), dtype=cp.float64)
    v2v_interference = cp.zeros((n_vehicles * 3,), dtype=cp.float64)

    if int(links.size) > 0:
        rb = rb_flat[links]
        tx = veh_idx[links]
        rx = dest_flat[links]
        p_db = power_db[cp.clip(power_flat[links], 0, len(power_db_list) - 1)]

        v2i_weights = 10.0 ** (
            (
                p_db
                - v2i_ch[tx, rb]
                + float(veh_ant_gain)
                + float(bs_ant_gain)
                - float(bs_noise_figure)
            )
            / 10.0
        )
        cp.add.at(v2i_interference, rb, v2i_weights)

        v2v_signal[links] = 10.0 ** (
            (
                p_db
                - v2v_ch[tx, rx, rb]
                + 2.0 * float(veh_ant_gain)
                - float(veh_noise_figure)
            )
            / 10.0
        )

        if int(links.size) < _grouped_min_links():
            pair_power = 10.0 ** (
                (
                    p_db[:, None]
                    - v2v_ch[tx[:, None], rx[None, :], rb[None, :]]
                    + 2.0 * float(veh_ant_gain)
                    - float(veh_noise_figure)
                )
                / 10.0
            )
            pair_power *= (rb[:, None] == rb[None, :]) & (links[:, None] != links[None, :])
            v2v_interference[links] = cp.sum(pair_power, axis=0)
            if include_v2i_to_v2v:
                v2i_mask = rb < n_vehicles
                if bool(cp.any(v2i_mask).item()):
                    v2i_links = links[v2i_mask]
                    v2i_rb = rb[v2i_mask]
                    v2i_rx = rx[v2i_mask]
                    v2v_interference[v2i_links] += 10.0 ** (
                        (
                            float(v2i_power_db)
                            - v2v_ch[v2i_rb, v2i_rx, v2i_rb]
                            + 2.0 * float(veh_ant_gain)
                            - float(veh_noise_figure)
                        )
                        / 10.0
                    )
        else:
            for rb_value in range(n_rb):
                rb_mask = rb == rb_value
                if not bool(cp.any(rb_mask).item()):
                    continue
                rb_links = links[rb_mask]
                rb_tx = tx[rb_mask]
                rb_rx = rx[rb_mask]
                rb_p_db = p_db[rb_mask]
                pair_power = 10.0 ** (
                    (
                        rb_p_db[:, None]
                        - v2v_ch[rb_tx[:, None], rb_rx[None, :], rb_value]
                        + 2.0 * float(veh_ant_gain)
                        - float(veh_noise_figure)
                    )
                    / 10.0
                )
                if int(pair_power.shape[0]) == int(pair_power.shape[1]):
                    cp.fill_diagonal(pair_power, 0.0)
                v2v_interference[rb_links] = cp.sum(pair_power, axis=0)
                if include_v2i_to_v2v and rb_value < n_vehicles:
                    v2v_interference[rb_links] += 10.0 ** (
                        (
                            float(v2i_power_db)
                            - v2v_ch[rb_value, rb_rx, rb_value]
                            + 2.0 * float(veh_ant_gain)
                            - float(veh_noise_figure)
                        )
                        / 10.0
                    )

    v2i_interference = v2i_interference + float(sig2)
    v2v_interference = v2v_interference.reshape(n_vehicles, 3) + float(sig2)
    v2v_signal = v2v_signal.reshape(n_vehicles, 3)
    v2v_rate = cp.log2(1.0 + v2v_signal / cp.maximum(v2v_interference, 1e-12))

    rb_count = min(n_rb, n_vehicles)
    v2i_signals = (
        float(v2i_power_db)
        - v2i_abs[:rb_count]
        + float(veh_ant_gain)
        + float(bs_ant_gain)
        - float(bs_noise_figure)
    )
    v2i_rate = cp.log2(1.0 + 10.0 ** (v2i_signals / 10.0) / v2i_interference[:rb_count])

    return cp.asnumpy(v2v_rate), cp.asnumpy(v2i_rate), cp.asnumpy(v2i_interference), cp.asnumpy(v2v_interference)


def compute_training_batch_rates_cupy(
    actions_power: np.ndarray,
    destinations: np.ndarray,
    v2v_channels_with_fastfading: np.ndarray,
    v2i_channels_with_fastfading: np.ndarray,
    v2i_channels_abs: np.ndarray,
    *,
    power_db_list: list[float] | np.ndarray,
    v2i_power_db: float,
    veh_ant_gain: float,
    bs_ant_gain: float,
    bs_noise_figure: float,
    veh_noise_figure: float,
    sig2: float,
    n_rb: int,
    n_vehicles: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """CuPy equivalent of tensorized_comm.compute_training_batch_rates."""

    v2v_rate, v2i_rate, v2i_interference, _v2v_interference = _compute_rates_cupy(
        actions_power,
        destinations,
        v2v_channels_with_fastfading,
        v2i_channels_with_fastfading,
        v2i_channels_abs,
        power_db_list=power_db_list,
        v2i_power_db=v2i_power_db,
        veh_ant_gain=veh_ant_gain,
        bs_ant_gain=bs_ant_gain,
        bs_noise_figure=bs_noise_figure,
        veh_noise_figure=veh_noise_figure,
        sig2=sig2,
        n_rb=n_rb,
        n_vehicles=n_vehicles,
    )
    return v2v_rate, v2i_rate, v2i_interference


def compute_async_rates_cupy(
    actions_power: np.ndarray,
    active_links: np.ndarray,
    destinations: np.ndarray,
    v2v_channels_with_fastfading: np.ndarray,
    v2i_channels_with_fastfading: np.ndarray,
    v2i_channels_abs: np.ndarray,
    *,
    power_db_list: list[float] | np.ndarray,
    v2i_power_db: float,
    veh_ant_gain: float,
    bs_ant_gain: float,
    bs_noise_figure: float,
    veh_noise_figure: float,
    sig2: float,
    n_rb: int,
    n_vehicles: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """CuPy equivalent of tensorized_comm.compute_async_rates."""

    v2v_rate, v2i_rate, v2i_interference, v2v_interference = _compute_rates_cupy(
        actions_power,
        destinations,
        v2v_channels_with_fastfading,
        v2i_channels_with_fastfading,
        v2i_channels_abs,
        power_db_list=power_db_list,
        v2i_power_db=v2i_power_db,
        veh_ant_gain=veh_ant_gain,
        bs_ant_gain=bs_ant_gain,
        bs_noise_figure=bs_noise_figure,
        veh_noise_figure=veh_noise_figure,
        sig2=sig2,
        n_rb=n_rb,
        n_vehicles=n_vehicles,
        active_links=active_links,
        include_v2i_to_v2v=True,
    )
    return v2i_rate, v2v_rate, v2v_interference, v2i_interference


def compute_v2v_interference_db_cupy(
    actions_power: np.ndarray,
    destinations: np.ndarray,
    v2v_channels_with_fastfading: np.ndarray,
    *,
    power_db_list: list[float] | np.ndarray,
    veh_ant_gain: float,
    veh_noise_figure: float,
    sig2: float,
    n_rb: int,
) -> np.ndarray:
    """CuPy equivalent of tensorized_comm.compute_v2v_interference_db."""

    if not HAS_CUPY:
        raise RuntimeError("CuPy backend requested but cupy is unavailable")
    actions = cp.asarray(actions_power[:, :, 0], dtype=cp.int32)
    powers = cp.asarray(actions_power[:, :, 1], dtype=cp.int32)
    destinations = cp.asarray(destinations[:, :3], dtype=cp.int32)
    v2v_ch = cp.asarray(v2v_channels_with_fastfading, dtype=cp.float64)
    power_db = cp.asarray(power_db_list, dtype=cp.float64)
    n_vehicles = int(actions.shape[0])
    n_rb = int(n_rb)
    if n_vehicles == 0:
        return np.zeros((0, 3, n_rb), dtype=np.float32)

    veh_idx = cp.repeat(cp.arange(n_vehicles, dtype=cp.int32), 3)
    rb_flat = actions.reshape(-1)
    power_flat = powers.reshape(-1)
    dest_flat = destinations.reshape(-1)
    valid = (rb_flat >= 0) & (rb_flat < n_rb) & (dest_flat >= 0) & (dest_flat < n_vehicles)
    links = cp.nonzero(valid)[0]
    interference = cp.zeros((n_vehicles * 3, n_rb), dtype=cp.float64) + float(sig2)
    if int(links.size) > 0:
        rb = rb_flat[links]
        tx = veh_idx[links]
        rx = dest_flat[links]
        p_db = power_db[cp.clip(power_flat[links], 0, len(power_db_list) - 1)]
        if int(links.size) < _grouped_min_links():
            pair_power = 10.0 ** (
                (
                    p_db[:, None]
                    - v2v_ch[tx[:, None], rx[None, :], rb[None, :]]
                    + 2.0 * float(veh_ant_gain)
                    - float(veh_noise_figure)
                )
                / 10.0
            )
            pair_power *= (rb[:, None] == rb[None, :]) & (links[:, None] != links[None, :])
            interference[links, rb] += cp.sum(pair_power, axis=0)
        else:
            for rb_value in range(n_rb):
                rb_mask = rb == rb_value
                if not bool(cp.any(rb_mask).item()):
                    continue
                rb_links = links[rb_mask]
                rb_tx = tx[rb_mask]
                rb_rx = rx[rb_mask]
                rb_p_db = p_db[rb_mask]
                pair_power = 10.0 ** (
                    (
                        rb_p_db[:, None]
                        - v2v_ch[rb_tx[:, None], rb_rx[None, :], rb_value]
                        + 2.0 * float(veh_ant_gain)
                        - float(veh_noise_figure)
                    )
                    / 10.0
                )
                if int(pair_power.shape[0]) == int(pair_power.shape[1]):
                    cp.fill_diagonal(pair_power, 0.0)
                interference[rb_links, rb_value] += cp.sum(pair_power, axis=0)
    return cp.asnumpy(10.0 * cp.log10(interference).reshape(n_vehicles, 3, n_rb))
