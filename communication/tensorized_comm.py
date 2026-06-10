"""Vectorized communication performance kernels for the tensorized experiment."""

from __future__ import annotations

import numpy as np


def _flat_link_arrays(n_vehicles: int) -> tuple[np.ndarray, np.ndarray]:
    veh_idx = np.repeat(np.arange(n_vehicles, dtype=np.int32), 3)
    link_idx = np.tile(np.arange(3, dtype=np.int32), n_vehicles)
    return veh_idx, link_idx


def _v2v_signal_interference_flat(
    *,
    rb_flat: np.ndarray,
    power_flat: np.ndarray,
    dest_flat: np.ndarray,
    veh_idx: np.ndarray,
    v2v_channels_with_fastfading: np.ndarray,
    power_db: np.ndarray,
    veh_ant_gain: float,
    veh_noise_figure: float,
    n_rb: int,
    n_vehicles: int,
    active_flat: np.ndarray | None = None,
    include_v2i_to_v2v: bool = False,
    v2i_power_db: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return flat V2V signal/interference for valid links without per-RB loops."""

    rb_flat = np.asarray(rb_flat, dtype=np.int32).reshape(-1)
    power_flat = np.asarray(power_flat, dtype=np.int32).reshape(-1)
    dest_flat = np.asarray(dest_flat, dtype=np.int32).reshape(-1)
    if active_flat is None:
        active_flat = np.ones(rb_flat.shape, dtype=bool)
    else:
        active_flat = np.asarray(active_flat, dtype=bool).reshape(-1)

    valid = (
        active_flat
        & (rb_flat >= 0)
        & (rb_flat < int(n_rb))
        & (dest_flat >= 0)
        & (dest_flat < int(n_vehicles))
    )
    links = np.flatnonzero(valid)
    signal = np.zeros(rb_flat.shape, dtype=np.float64)
    interference = np.zeros(rb_flat.shape, dtype=np.float64)
    if links.size == 0:
        return signal, interference, links

    rb = rb_flat[links]
    tx = veh_idx[links]
    rx = dest_flat[links]
    p_db = power_db[np.clip(power_flat[links], 0, len(power_db) - 1)].astype(np.float64, copy=False)

    signal[links] = 10.0 ** (
        (
            p_db
            - v2v_channels_with_fastfading[tx, rx, rb]
            + 2.0 * float(veh_ant_gain)
            - float(veh_noise_figure)
        )
        / 10.0
    )

    same_rb = rb[:, None] == rb[None, :]
    not_self = links[:, None] != links[None, :]
    pair_channels = v2v_channels_with_fastfading[tx[:, None], rx[None, :], rb[None, :]]
    pair_power = 10.0 ** (
        (
            p_db[:, None]
            - pair_channels
            + 2.0 * float(veh_ant_gain)
            - float(veh_noise_figure)
        )
        / 10.0
    )
    interference[links] = np.sum(pair_power * (same_rb & not_self), axis=0)

    if include_v2i_to_v2v:
        v2i_mask = rb < int(n_vehicles)
        if np.any(v2i_mask):
            v2i_links = links[v2i_mask]
            v2i_rb = rb[v2i_mask]
            v2i_rx = rx[v2i_mask]
            interference[v2i_links] += 10.0 ** (
                (
                    float(v2i_power_db)
                    - v2v_channels_with_fastfading[v2i_rb, v2i_rx, v2i_rb]
                    + 2.0 * float(veh_ant_gain)
                    - float(veh_noise_figure)
                )
                / 10.0
            )

    return signal, interference, links


def _v2v_signal_interference_grouped(
    *,
    rb_flat: np.ndarray,
    power_flat: np.ndarray,
    dest_flat: np.ndarray,
    veh_idx: np.ndarray,
    v2v_channels_with_fastfading: np.ndarray,
    power_db: np.ndarray,
    veh_ant_gain: float,
    veh_noise_figure: float,
    n_rb: int,
    n_vehicles: int,
    active_flat: np.ndarray | None = None,
    include_v2i_to_v2v: bool = False,
    v2i_power_db: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return flat V2V signal/interference using per-RB grouped matrix blocks."""

    rb_flat = np.asarray(rb_flat, dtype=np.int32).reshape(-1)
    power_flat = np.asarray(power_flat, dtype=np.int32).reshape(-1)
    dest_flat = np.asarray(dest_flat, dtype=np.int32).reshape(-1)
    if active_flat is None:
        active_flat = np.ones(rb_flat.shape, dtype=bool)
    else:
        active_flat = np.asarray(active_flat, dtype=bool).reshape(-1)

    valid = (
        active_flat
        & (rb_flat >= 0)
        & (rb_flat < int(n_rb))
        & (dest_flat >= 0)
        & (dest_flat < int(n_vehicles))
    )
    all_links = np.flatnonzero(valid)
    signal = np.zeros(rb_flat.shape, dtype=np.float64)
    interference = np.zeros(rb_flat.shape, dtype=np.float64)
    if all_links.size == 0:
        return signal, interference, all_links

    for rb in range(int(n_rb)):
        links = all_links[rb_flat[all_links] == rb]
        if links.size == 0:
            continue
        tx = veh_idx[links]
        rx = dest_flat[links]
        p_db = power_db[np.clip(power_flat[links], 0, len(power_db) - 1)].astype(np.float64, copy=False)
        signal[links] = 10.0 ** (
            (
                p_db
                - v2v_channels_with_fastfading[tx, rx, rb]
                + 2.0 * float(veh_ant_gain)
                - float(veh_noise_figure)
            )
            / 10.0
        )
        contrib = 10.0 ** (
            (
                p_db[:, None]
                - v2v_channels_with_fastfading[tx[:, None], rx[None, :], rb]
                + 2.0 * float(veh_ant_gain)
                - float(veh_noise_figure)
            )
            / 10.0
        )
        if contrib.shape[0] == contrib.shape[1]:
            np.fill_diagonal(contrib, 0.0)
        interference[links] += np.sum(contrib, axis=0)

        if include_v2i_to_v2v and rb < int(n_vehicles):
            interference[links] += 10.0 ** (
                (
                    float(v2i_power_db)
                    - v2v_channels_with_fastfading[rb, rx, rb]
                    + 2.0 * float(veh_ant_gain)
                    - float(veh_noise_figure)
                )
                / 10.0
            )

    return signal, interference, all_links


def _v2v_signal_interference_auto(**kwargs) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rb_flat = np.asarray(kwargs["rb_flat"]).reshape(-1)
    active_flat = kwargs.get("active_flat")
    if active_flat is None:
        n_active = int(np.sum((rb_flat >= 0) & (rb_flat < int(kwargs["n_rb"]))))
    else:
        active_arr = np.asarray(active_flat, dtype=bool).reshape(-1)
        n_active = int(np.sum(active_arr & (rb_flat >= 0) & (rb_flat < int(kwargs["n_rb"]))))
    if n_active <= 160:
        return _v2v_signal_interference_flat(**kwargs)
    return _v2v_signal_interference_grouped(**kwargs)


def compute_v2v_interference_db(
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
    """Compute V2V interference tensor in dB for all receiver links and RBs."""

    actions = np.asarray(actions_power[:, :, 0], dtype=np.int32)
    powers = np.asarray(actions_power[:, :, 1], dtype=np.int32)
    destinations = np.asarray(destinations, dtype=np.int32)
    n_vehicles = int(actions.shape[0])
    if n_vehicles == 0:
        return np.zeros((0, 3, n_rb), dtype=np.float32)

    veh_idx, _link_idx = _flat_link_arrays(n_vehicles)
    rb_flat = actions.reshape(-1)
    power_flat = powers.reshape(-1)
    dest_flat = destinations[:, :3].reshape(-1)
    power_db = np.asarray(power_db_list, dtype=np.float32)
    interference = np.zeros((n_vehicles * 3, n_rb), dtype=np.float64) + float(sig2)
    _signal, raw_interference, links = _v2v_signal_interference_auto(
        rb_flat=rb_flat,
        power_flat=power_flat,
        dest_flat=dest_flat,
        veh_idx=veh_idx,
        v2v_channels_with_fastfading=v2v_channels_with_fastfading,
        power_db=power_db,
        veh_ant_gain=veh_ant_gain,
        veh_noise_figure=veh_noise_figure,
        n_rb=n_rb,
        n_vehicles=n_vehicles,
    )
    if links.size:
        interference[links, rb_flat[links]] += raw_interference[links]

    return (10.0 * np.log10(interference)).reshape(n_vehicles, 3, n_rb)


def compute_async_rates(
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
    """Compute V2I rates plus V2V rates/interference for async communication."""

    actions = np.asarray(actions_power[:, :, 0], dtype=np.int32)
    powers = np.asarray(actions_power[:, :, 1], dtype=np.int32)
    active_links = np.asarray(active_links, dtype=bool)
    destinations = np.asarray(destinations, dtype=np.int32)
    power_db = np.asarray(power_db_list, dtype=np.float32)
    veh_idx, _link_idx = _flat_link_arrays(int(n_vehicles))
    rb_flat = actions.reshape(-1)
    power_flat = powers.reshape(-1)
    active_flat = active_links.reshape(-1)
    dest_flat = destinations[:, :3].reshape(-1)

    valid_active = active_flat & (rb_flat >= 0) & (rb_flat < int(n_rb))
    active_links_flat = np.flatnonzero(valid_active)
    active_rb = rb_flat[active_links_flat]
    active_tx = veh_idx[active_links_flat]
    active_power = power_db[np.clip(power_flat[active_links_flat], 0, len(power_db) - 1)]

    if active_links_flat.size:
        v2i_weights = 10.0 ** (
            (
                active_power
                - v2i_channels_with_fastfading[active_tx, active_rb]
                + float(veh_ant_gain)
                + float(bs_ant_gain)
                - float(bs_noise_figure)
            )
            / 10.0
        )
        v2i_interference = np.bincount(active_rb, weights=v2i_weights, minlength=int(n_rb)).astype(np.float64)
    else:
        v2i_interference = np.zeros((int(n_rb),), dtype=np.float64)
    v2i_interference = v2i_interference + float(sig2)

    v2v_signal, v2v_interference, _links = _v2v_signal_interference_auto(
        rb_flat=rb_flat,
        power_flat=power_flat,
        dest_flat=dest_flat,
        veh_idx=veh_idx,
        v2v_channels_with_fastfading=v2v_channels_with_fastfading,
        power_db=power_db,
        veh_ant_gain=veh_ant_gain,
        veh_noise_figure=veh_noise_figure,
        n_rb=n_rb,
        n_vehicles=n_vehicles,
        active_flat=active_flat,
        include_v2i_to_v2v=True,
        v2i_power_db=v2i_power_db,
    )

    v2v_interference = v2v_interference.reshape(int(n_vehicles), 3) + float(sig2)
    v2v_signal = v2v_signal.reshape(int(n_vehicles), 3)
    v2v_rate = np.log2(1.0 + v2v_signal / v2v_interference)

    rb_count = min(int(n_rb), int(n_vehicles))
    v2i_signals = (
        float(v2i_power_db)
        - np.asarray(v2i_channels_abs[:rb_count], dtype=np.float64)
        + float(veh_ant_gain)
        + float(bs_ant_gain)
        - float(bs_noise_figure)
    )
    v2i_rate = np.log2(1.0 + 10.0 ** (v2i_signals / 10.0) / v2i_interference[:rb_count])
    return v2i_rate, v2v_rate, v2v_interference, v2i_interference


def compute_training_batch_rates(
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
    """Compute the rate core used by Environment.batch_reward_all."""

    actions = np.asarray(actions_power[:, :, 0], dtype=np.int32)
    powers = np.asarray(actions_power[:, :, 1], dtype=np.int32)
    destinations = np.asarray(destinations, dtype=np.int32)
    power_db = np.asarray(power_db_list, dtype=np.float32)
    veh_idx, _link_idx = _flat_link_arrays(int(n_vehicles))
    rb_flat = actions.reshape(-1)
    power_flat = powers.reshape(-1)
    dest_flat = destinations[:, :3].reshape(-1)
    valid = (rb_flat >= 0) & (rb_flat < int(n_rb))
    links_flat = np.flatnonzero(valid)
    link_rb = rb_flat[links_flat]
    link_tx = veh_idx[links_flat]
    link_power = power_db[np.clip(power_flat[links_flat], 0, len(power_db) - 1)]

    if links_flat.size:
        v2i_weights = 10.0 ** (
            (
                link_power
                - v2i_channels_with_fastfading[link_tx, link_rb]
                + float(veh_ant_gain)
                + float(bs_ant_gain)
                - float(bs_noise_figure)
            )
            / 10.0
        )
        v2i_interference = np.bincount(link_rb, weights=v2i_weights, minlength=int(n_rb)).astype(np.float64)
    else:
        v2i_interference = np.zeros((int(n_rb),), dtype=np.float64)
    v2i_interference = v2i_interference + float(sig2)

    v2v_signal, v2v_interference, _links = _v2v_signal_interference_auto(
        rb_flat=rb_flat,
        power_flat=power_flat,
        dest_flat=dest_flat,
        veh_idx=veh_idx,
        v2v_channels_with_fastfading=v2v_channels_with_fastfading,
        power_db=power_db,
        veh_ant_gain=veh_ant_gain,
        veh_noise_figure=veh_noise_figure,
        n_rb=n_rb,
        n_vehicles=n_vehicles,
    )

    v2v_interference = v2v_interference.reshape(int(n_vehicles), 3) + float(sig2)
    v2v_signal = v2v_signal.reshape(int(n_vehicles), 3)
    v2v_rate = np.log2(1.0 + v2v_signal / np.maximum(v2v_interference, 1e-12))

    rb_count = min(int(n_rb), int(n_vehicles))
    v2i_signals = (
        float(v2i_power_db)
        - np.asarray(v2i_channels_abs[:rb_count], dtype=np.float64)
        + float(veh_ant_gain)
        + float(bs_ant_gain)
        - float(bs_noise_figure)
    )
    v2i_rate = np.log2(1.0 + 10.0 ** (v2i_signals / 10.0) / v2i_interference[:rb_count])
    return v2v_rate, v2i_rate, v2i_interference


def compute_single_link_reward_table(
    *,
    tx: int,
    link: int,
    rx: int,
    demand_value: float,
    individual_time_limit_value: float,
    v2v_channels_with_fastfading: np.ndarray,
    v2i_channels_with_fastfading: np.ndarray,
    v2i_channels_abs: np.ndarray,
    power_db_list: list[float] | np.ndarray,
    v2i_power_db: float,
    veh_ant_gain: float,
    bs_ant_gain: float,
    bs_noise_figure: float,
    veh_noise_figure: float,
    sig2: float,
    n_rb: int,
    n_vehicles: int,
    v2v_limit: float,
) -> tuple[np.ndarray, float, float]:
    """Vectorized equivalent of Environment.Compute_Performance_Reward_Batch."""

    del link
    rb = np.arange(int(n_rb), dtype=np.int32)
    p_db = np.asarray(power_db_list, dtype=np.float64)
    tx = int(tx)
    rx = int(rx)
    rx = int(np.clip(rx, 0, max(0, int(n_vehicles) - 1)))

    v2v_signal = 10.0 ** (
        (
            p_db[None, :]
            - v2v_channels_with_fastfading[tx, rx, rb][:, None]
            + 2.0 * float(veh_ant_gain)
            - float(veh_noise_figure)
        )
        / 10.0
    )
    rb_vehicle = rb % max(1, int(n_vehicles))
    interf_v2v = 10.0 ** (
        (
            float(v2i_power_db)
            - v2v_channels_with_fastfading[rb_vehicle, rx, rb][:, None]
            + 2.0 * float(veh_ant_gain)
            - float(veh_noise_figure)
        )
        / 10.0
    )
    v2v_rate = np.log2(1.0 + v2v_signal / (interf_v2v + float(sig2)))

    v2i_signal = 10.0 ** (
        (
            float(v2i_power_db)
            + float(veh_ant_gain)
            + float(bs_ant_gain)
            - float(bs_noise_figure)
            - v2i_channels_abs[np.minimum(rb, max(0, int(n_vehicles) - 1))][:, None]
        )
        / 10.0
    )
    extra_i = 10.0 ** (
        (
            p_db[None, :]
            - v2i_channels_with_fastfading[tx, rb][:, None]
            + float(veh_ant_gain)
            + float(bs_ant_gain)
            - float(bs_noise_figure)
        )
        / 10.0
    )
    v2i_rate = np.log2(1.0 + v2i_signal / (float(sig2) + extra_i))

    combo = 0.1 * np.tanh(v2i_rate / 50.0) + 0.9 * np.tanh(v2v_rate / 50.0)
    if float(demand_value) < 0.0:
        time_left = float(v2v_limit)
    else:
        time_left = float(individual_time_limit_value)
    penalty = (float(v2v_limit) - time_left) / float(v2v_limit)
    return combo.astype(np.float64, copy=False), -float(penalty), time_left
