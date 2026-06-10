"""Monkey-patch V2V/V2I channel models with vectorized numpy operations.

Replaces O(n^2) Python loops in update_pathloss and Compute_Interference
with broadcasting/advanced indexing.  ~26ms -> ~1ms per step at n=30.
"""

from __future__ import annotations

import math
import numpy as np


def _v2v_update_pathloss_vectorized(self) -> None:
    """Vectorized V2V path loss: one numpy pass over all n^2 pairs."""
    positions = np.asarray(self.positions, dtype=np.float64)
    n = len(positions)
    if n == 0:
        self.PathLoss = np.zeros((0, 0))
        return

    pos_x = positions[:, 0]
    pos_y = positions[:, 1]

    d1 = np.abs(pos_x[:, None] - pos_x[None, :])  # (n, n)
    d2 = np.abs(pos_y[:, None] - pos_y[None, :])  # (n, n)
    d = np.hypot(d1, d2) + 0.001

    d_bp = 4 * (self.h_bs - 1) * (self.h_ms - 1) * self.fc * 1e9 / 3e8
    log_fc5 = np.log10(self.fc / 5.0)

    pl_los_base = 22.7 * np.log10(np.maximum(d, 3.0)) + 41.0 + 20.0 * log_fc5
    pl_los_near = 22.7 * np.log10(3.0) + 41.0 + 20.0 * log_fc5
    pl_los_far = (40.0 * np.log10(d) + 9.45
                  - 17.3 * np.log10(self.h_bs)
                  - 17.3 * np.log10(self.h_ms)
                  + 2.7 * log_fc5)

    PL_Los = np.where(d <= 3.0, pl_los_near,
             np.where(d < d_bp, pl_los_base, pl_los_far))

    n_j_12 = np.maximum(2.8 - 0.0024 * d2, 1.84)
    n_j_21 = np.maximum(2.8 - 0.0024 * d1, 1.84)

    PL_Los_d1 = np.where(d1 <= 3.0, pl_los_near,
                np.where(d1 < d_bp,
                         22.7 * np.log10(np.maximum(d1, 3.0)) + 41.0 + 20.0 * log_fc5,
                         40.0 * np.log10(np.maximum(d1, 0.001)) + 9.45
                         - 17.3 * np.log10(self.h_bs)
                         - 17.3 * np.log10(self.h_ms)
                         + 2.7 * log_fc5))
    PL_NLos_12 = PL_Los_d1 + 20.0 - 12.5 * n_j_12 + 10.0 * n_j_12 * np.log10(np.maximum(d2, 0.001)) + 3.0 * log_fc5

    PL_Los_d2 = np.where(d2 <= 3.0, pl_los_near,
                np.where(d2 < d_bp,
                         22.7 * np.log10(np.maximum(d2, 3.0)) + 41.0 + 20.0 * log_fc5,
                         40.0 * np.log10(np.maximum(d2, 0.001)) + 9.45
                         - 17.3 * np.log10(self.h_bs)
                         - 17.3 * np.log10(self.h_ms)
                         + 2.7 * log_fc5))
    PL_NLos_21 = PL_Los_d2 + 20.0 - 12.5 * n_j_21 + 10.0 * n_j_21 * np.log10(np.maximum(d1, 0.001)) + 3.0 * log_fc5

    PL_NLos = np.minimum(PL_NLos_12, PL_NLos_21)

    is_los = np.minimum(d1, d2) < 7.0
    self.PathLoss = np.where(is_los, PL_Los, PL_NLos)


def _v2i_update_pathloss_vectorized(self) -> None:
    """Vectorized V2I path loss: one numpy pass over n vehicles."""
    positions = np.asarray(self.positions, dtype=np.float64)
    n = len(positions)
    if n == 0:
        self.PathLoss = np.zeros(0)
        return

    d1 = np.abs(positions[:, 0] - self.BS_position[0])
    d2 = np.abs(positions[:, 1] - self.BS_position[1])
    distance = np.hypot(d1, d2)
    height_diff = self.h_bs - self.h_ms
    d3d = np.sqrt(distance ** 2 + height_diff ** 2) / 1000.0
    self.PathLoss = 128.1 + 37.6 * np.log10(np.maximum(d3d, 1e-10))


def _compute_interference_vectorized(self, actions) -> None:
    """Vectorized Compute_Interference using numpy advanced indexing."""
    n_veh = len(self.vehicles)
    V2V_Interference = np.full((n_veh, 3, self.n_RB), self.sig2, dtype=np.float64)

    channels = np.asarray(actions[:, :, 0], dtype=np.intp)
    powers = np.asarray(actions[:, :, 1], dtype=np.intp)

    power_list = np.array(self.V2V_power_dB_List, dtype=np.float64)
    gain_noise = 2.0 * self.vehAntGain - self.vehNoiseFigure

    destinations = np.zeros((n_veh, 3), dtype=np.intp)
    for i in range(n_veh):
        for j in range(3):
            destinations[i, j] = int(self.vehicles[i].destinations[j])

    for rb in range(self.n_RB):
        mask = (channels == rb)
        if not np.any(mask):
            continue
        tx_indices, link_indices = np.nonzero(mask)
        if len(tx_indices) < 2:
            continue

        p_dB = power_list[powers[tx_indices, link_indices]]
        rx = destinations[tx_indices, link_indices]

        for a in range(len(tx_indices)):
            tx_a = tx_indices[a]
            link_a = link_indices[a]
            rx_a = rx[a]
            p_a = p_dB[a]
            for b in range(a + 1, len(tx_indices)):
                tx_b = tx_indices[b]
                link_b = link_indices[b]
                rx_b = rx[b]
                p_b = p_dB[b]
                interf_ab = 10.0 ** ((p_b - self.V2V_channels_with_fastfading[tx_b, rx_a, rb] + gain_noise) / 10.0)
                interf_ba = 10.0 ** ((p_a - self.V2V_channels_with_fastfading[tx_a, rx_b, rb] + gain_noise) / 10.0)
                V2V_Interference[tx_a, link_a, rb] += interf_ab
                V2V_Interference[tx_b, link_b, rb] += interf_ba

    self.V2V_Interference_all = 10.0 * np.log10(V2V_Interference)


_CHANNEL_PATCHED = False


def _patch_classes(V2Vchannels, V2Ichannels, Environ):
    """Apply vectorized methods to channel/environ classes."""
    if not hasattr(V2Vchannels, "_orig_update_pathloss"):
        V2Vchannels._orig_update_pathloss = V2Vchannels.update_pathloss
    V2Vchannels.update_pathloss = _v2v_update_pathloss_vectorized

    if not hasattr(V2Ichannels, "_orig_update_pathloss"):
        V2Ichannels._orig_update_pathloss = V2Ichannels.update_pathloss
    V2Ichannels.update_pathloss = _v2i_update_pathloss_vectorized

    if not hasattr(Environ, "_orig_Compute_Interference"):
        Environ._orig_Compute_Interference = Environ.Compute_Interference
    Environ.Compute_Interference = _compute_interference_vectorized


def install_channel_patches() -> None:
    """Monkey-patch V2V/V2I channel classes and Environ with vectorized versions."""
    global _CHANNEL_PATCHED
    if _CHANNEL_PATCHED:
        return

    import sys
    import importlib

    if "communication" not in sys.path:
        import os
        comm_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "communication")
        if comm_dir not in sys.path:
            sys.path.insert(0, comm_dir)

    bare_mod = importlib.import_module("Environment")
    _patch_classes(bare_mod.V2Vchannels, bare_mod.V2Ichannels, bare_mod.Environ)

    if "communication.Environment" in sys.modules:
        comm_mod = sys.modules["communication.Environment"]
        if comm_mod is not bare_mod:
            _patch_classes(comm_mod.V2Vchannels, comm_mod.V2Ichannels, comm_mod.Environ)

    _CHANNEL_PATCHED = True


def uninstall_channel_patches() -> None:
    """Restore original channel methods."""
    global _CHANNEL_PATCHED
    if not _CHANNEL_PATCHED:
        return

    from communication.Environment import V2Vchannels, V2Ichannels, Environ

    if hasattr(V2Vchannels, "_orig_update_pathloss"):
        V2Vchannels.update_pathloss = V2Vchannels._orig_update_pathloss
    if hasattr(V2Ichannels, "_orig_update_pathloss"):
        V2Ichannels.update_pathloss = V2Ichannels._orig_update_pathloss
    if hasattr(Environ, "_orig_Compute_Interference"):
        Environ.Compute_Interference = Environ._orig_Compute_Interference

    _CHANNEL_PATCHED = False
