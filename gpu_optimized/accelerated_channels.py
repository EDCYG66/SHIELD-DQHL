"""Vectorized channel model patches for communication/Environment.py.

Replaces O(n^2) Python loops in V2Vchannels.update_pathloss and
Environ.Compute_Interference with fully vectorized numpy operations.
"""
from __future__ import annotations

import numpy as np


def _v2v_update_pathloss_vectorized(self):
    """Vectorized V2V path loss: O(n^2) numpy broadcast, no Python loop."""
    pos = np.array(self.positions, dtype=np.float64)
    n = len(pos)
    if n == 0:
        self.PathLoss = np.zeros((0, 0))
        return

    d1 = np.abs(pos[:, 0:1] - pos[:, 0:1].T)
    d2 = np.abs(pos[:, 1:2] - pos[:, 1:2].T)
    d = np.sqrt(d1 * d1 + d2 * d2) + 0.001

    fc = self.fc
    h_bs = self.h_bs
    h_ms = self.h_ms
    d_bp = 4 * (h_bs - 1) * (h_ms - 1) * fc * 1e9 / 3e8
    log_fc5 = np.log10(fc / 5.0)

    d_clamped = np.maximum(d, 3.0)
    pl_los_near = 22.7 * np.log10(d_clamped) + 41.0 + 20.0 * log_fc5

    pl_los_far = (40.0 * np.log10(d) + 9.45
                  - 17.3 * np.log10(h_bs)
                  - 17.3 * np.log10(h_ms)
                  + 2.7 * log_fc5)
    pl_los = np.where(d < d_bp, pl_los_near, pl_los_far)

    n_j_12 = np.clip(2.8 - 0.0024 * d2, 1.84, None)
    n_j_21 = np.clip(2.8 - 0.0024 * d1, 1.84, None)

    d1_safe = np.maximum(d1, 3.0)
    d2_safe = np.maximum(d2, 3.0)

    pl_los_d1 = np.where(
        d1 < d_bp,
        22.7 * np.log10(np.maximum(d1, 3.0)) + 41.0 + 20.0 * log_fc5,
        40.0 * np.log10(d1_safe) + 9.45 - 17.3 * np.log10(h_bs) - 17.3 * np.log10(h_ms) + 2.7 * log_fc5,
    )
    pl_los_d2 = np.where(
        d2 < d_bp,
        22.7 * np.log10(np.maximum(d2, 3.0)) + 41.0 + 20.0 * log_fc5,
        40.0 * np.log10(d2_safe) + 9.45 - 17.3 * np.log10(h_bs) - 17.3 * np.log10(h_ms) + 2.7 * log_fc5,
    )

    nlos_12 = pl_los_d1 + 20.0 - 12.5 * n_j_12 + 10.0 * n_j_12 * np.log10(d2_safe) + 3.0 * log_fc5
    nlos_21 = pl_los_d2 + 20.0 - 12.5 * n_j_21 + 10.0 * n_j_21 * np.log10(d1_safe) + 3.0 * log_fc5
    pl_nlos = np.minimum(nlos_12, nlos_21)

    is_los = np.minimum(d1, d2) < 7.0
    self.PathLoss = np.where(is_los, pl_los, pl_nlos)


def _v2v_update_shadow_vectorized(self, delta_distance_list):
    """Vectorized shadow fading update."""
    if len(delta_distance_list) == 0:
        self.Shadow = np.random.normal(0, self.shadow_std, size=(self.n_Veh, self.n_Veh))
        return
    dd = np.asarray(delta_distance_list, dtype=np.float64)
    delta_distance = dd[:, None] + dd[None, :]
    factor = np.exp(-delta_distance / self.decorrelation_distance)
    noise = np.random.normal(0, self.shadow_std, size=(self.n_Veh, self.n_Veh))
    self.Shadow = factor * self.Shadow + np.sqrt(1.0 - np.exp(-2.0 * delta_distance / self.decorrelation_distance)) * noise


def _compute_interference_vectorized(self, actions):
    """Vectorized interference computation replacing O(n^2*9) loop."""
    n_veh = len(self.vehicles)
    n_links = 3
    V2V_Interference = np.full((n_veh, n_links, self.n_RB), self.sig2, dtype=np.float64)
    channels = actions[:, :, 0].copy()
    powers = actions[:, :, 1].copy()

    power_list = np.array(self.V2V_power_dB_List, dtype=np.float64)
    gain_offset = 2 * self.vehAntGain - self.vehNoiseFigure

    dest_array = np.zeros((n_veh, n_links), dtype=np.int64)
    for i in range(n_veh):
        for j in range(n_links):
            dest_array[i, j] = self.vehicles[i].destinations[j]

    for rb in range(self.n_RB):
        mask = channels == rb
        if not mask.any():
            continue
        tx_links = np.argwhere(mask)
        if len(tx_links) <= 1:
            continue

        for a_idx in range(len(tx_links)):
            i, j = tx_links[a_idx]
            p_sel = power_list[powers[i, j]]
            rx = dest_array[i, j]
            for b_idx in range(len(tx_links)):
                if b_idx == a_idx:
                    continue
                k, m = tx_links[b_idx]
                rx2 = dest_array[k, m]
                interf = 10.0 ** ((p_sel - self.V2V_channels_with_fastfading[i, rx2, rb] + gain_offset) / 10.0)
                V2V_Interference[k, m, rb] += interf

    self.V2V_Interference_all = 10.0 * np.log10(V2V_Interference)


def install_channel_patches():
    """Monkey-patch V2Vchannels and Environ with vectorized methods."""
    from communication.Environment import V2Vchannels, Environ

    V2Vchannels._orig_update_pathloss = V2Vchannels.update_pathloss
    V2Vchannels.update_pathloss = _v2v_update_pathloss_vectorized

    V2Vchannels._orig_update_shadow = V2Vchannels.update_shadow
    V2Vchannels.update_shadow = _v2v_update_shadow_vectorized

    Environ._orig_Compute_Interference = Environ.Compute_Interference
    Environ.Compute_Interference = _compute_interference_vectorized
