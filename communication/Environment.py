from __future__ import division
import numpy as np
import random
import math

try:
    from tensorized_backend import should_use_cupy
    from tensorized_comm import (
        compute_async_rates,
        compute_single_link_reward_table,
        compute_training_batch_rates,
        compute_v2v_interference_db,
    )
except Exception:  # pragma: no cover
    should_use_cupy = None
    compute_async_rates = None
    compute_single_link_reward_table = None
    compute_training_batch_rates = None
    compute_v2v_interference_db = None

try:
    from tensorized_comm_cupy import (
        compute_async_rates_cupy,
        compute_training_batch_rates_cupy,
        compute_v2v_interference_db_cupy,
    )
except Exception:  # pragma: no cover
    compute_async_rates_cupy = None
    compute_training_batch_rates_cupy = None
    compute_v2v_interference_db_cupy = None

# ==================== Channel Models ====================

class V2Vchannels:
    def __init__(self, n_Veh, n_RB):
        self.t = 0
        self.h_bs = 1.5
        self.h_ms = 1.5
        self.fc = 2
        self.decorrelation_distance = 10
        self.shadow_std = 3
        self.n_Veh = n_Veh
        self.n_RB = n_RB
        self.update_shadow([])

    def update_positions(self, positions):
        self.positions = positions

    def update_pathloss(self):
        positions = np.asarray(self.positions, dtype=np.float64)
        n = len(positions)
        if n == 0:
            self.PathLoss = np.zeros((0, 0), dtype=np.float64)
            return

        pos_x = positions[:, 0]
        pos_y = positions[:, 1]
        d1 = np.abs(pos_x[:, None] - pos_x[None, :])
        d2 = np.abs(pos_y[:, None] - pos_y[None, :])
        d = np.hypot(d1, d2) + 0.001

        d_bp = 4 * (self.h_bs - 1) * (self.h_ms - 1) * self.fc * 1e9 / 3e8
        log_fc5 = np.log10(self.fc / 5.0)
        pl_los_near = 22.7 * np.log10(3.0) + 41.0 + 20.0 * log_fc5
        pl_los_base = 22.7 * np.log10(np.maximum(d, 3.0)) + 41.0 + 20.0 * log_fc5
        pl_los_far = (
            40.0 * np.log10(d)
            + 9.45
            - 17.3 * np.log10(self.h_bs)
            - 17.3 * np.log10(self.h_ms)
            + 2.7 * log_fc5
        )
        pl_los = np.where(d <= 3.0, pl_los_near, np.where(d < d_bp, pl_los_base, pl_los_far))

        d1_safe = np.maximum(d1, 0.001)
        d2_safe = np.maximum(d2, 0.001)
        pl_los_d1 = np.where(
            d1 <= 3.0,
            pl_los_near,
            np.where(
                d1 < d_bp,
                22.7 * np.log10(np.maximum(d1, 3.0)) + 41.0 + 20.0 * log_fc5,
                40.0 * np.log10(d1_safe) + 9.45
                - 17.3 * np.log10(self.h_bs)
                - 17.3 * np.log10(self.h_ms)
                + 2.7 * log_fc5,
            ),
        )
        pl_los_d2 = np.where(
            d2 <= 3.0,
            pl_los_near,
            np.where(
                d2 < d_bp,
                22.7 * np.log10(np.maximum(d2, 3.0)) + 41.0 + 20.0 * log_fc5,
                40.0 * np.log10(d2_safe) + 9.45
                - 17.3 * np.log10(self.h_bs)
                - 17.3 * np.log10(self.h_ms)
                + 2.7 * log_fc5,
            ),
        )
        n_j_12 = np.maximum(2.8 - 0.0024 * d2, 1.84)
        n_j_21 = np.maximum(2.8 - 0.0024 * d1, 1.84)
        pl_nlos_12 = pl_los_d1 + 20.0 - 12.5 * n_j_12 + 10.0 * n_j_12 * np.log10(d2_safe) + 3.0 * log_fc5
        pl_nlos_21 = pl_los_d2 + 20.0 - 12.5 * n_j_21 + 10.0 * n_j_21 * np.log10(d1_safe) + 3.0 * log_fc5
        pl_nlos = np.minimum(pl_nlos_12, pl_nlos_21)

        self.PathLoss = np.where(np.minimum(d1, d2) < 7.0, pl_los, pl_nlos)

    def update_shadow(self, delta_distance_list):
        if len(delta_distance_list) == 0:
            self.Shadow = np.random.normal(0, self.shadow_std, size=(self.n_Veh, self.n_Veh))
            return
        n = len(delta_distance_list)
        delta = np.asarray(delta_distance_list, dtype=np.float64)
        delta_distance = delta[:, None] + delta[None, :]
        factor = np.exp(-delta_distance / self.decorrelation_distance)
        noise = np.random.normal(0, self.shadow_std, size=(self.n_Veh, self.n_Veh))
        self.Shadow = factor * self.Shadow + np.sqrt(1 - np.exp(-2 * delta_distance / self.decorrelation_distance)) * noise

    def update_fast_fading(self):
        h = (np.random.normal(size=(self.n_Veh, self.n_Veh, self.n_RB)) +
             1j * np.random.normal(size=(self.n_Veh, self.n_Veh, self.n_RB))) / np.sqrt(2)
        self.FastFading = 20 * np.log10(np.abs(h))

    def get_path_loss(self, position_A, position_B):
        d1 = abs(position_A[0] - position_B[0])
        d2 = abs(position_A[1] - position_B[1])
        d = math.hypot(d1, d2) + 0.001
        d_bp = 4 * (self.h_bs - 1) * (self.h_ms - 1) * self.fc * (10 ** 9) / (3 * 10 ** 8)

        def PL_Los(dv):
            if dv <= 3:
                return 22.7 * np.log10(3) + 41 + 20 * np.log10(self.fc / 5)
            if dv < d_bp:
                return 22.7 * np.log10(dv) + 41 + 20 * np.log10(self.fc / 5)
            return (40.0 * np.log10(dv) + 9.45 - 17.3 * np.log10(self.h_bs)
                    - 17.3 * np.log10(self.h_ms) + 2.7 * np.log10(self.fc / 5))

        def PL_NLos(d_a, d_b):
            n_j = max(2.8 - 0.0024 * d_b, 1.84)
            return PL_Los(d_a) + 20 - 12.5 * n_j + 10 * n_j * np.log10(d_b) + 3 * np.log10(self.fc / 5)

        if min(d1, d2) < 7:
            PL = PL_Los(d)
            self.ifLOS = True
            self.shadow_std = 3
        else:
            PL = min(PL_NLos(d1, d2), PL_NLos(d2, d1))
            self.ifLOS = False
            self.shadow_std = 4
        return PL


class V2Ichannels:
    def __init__(self, n_Veh, n_RB):
        self.h_bs = 25
        self.h_ms = 1.5
        self.Decorrelation_distance = 50
        self.BS_position = [750 / 2, 1299 / 2]
        self.shadow_std = 8
        self.n_Veh = n_Veh
        self.n_RB = n_RB
        self.update_shadow([])

    def update_positions(self, positions):
        self.positions = positions

    def update_pathloss(self):
        positions = np.asarray(self.positions, dtype=np.float64)
        n = len(positions)
        if n == 0:
            self.PathLoss = np.zeros(0, dtype=np.float64)
            return
        d1 = np.abs(positions[:, 0] - self.BS_position[0])
        d2 = np.abs(positions[:, 1] - self.BS_position[1])
        distance = np.hypot(d1, d2)
        height_diff = self.h_bs - self.h_ms
        d3d = np.sqrt(distance ** 2 + height_diff ** 2) / 1000.0
        self.PathLoss = 128.1 + 37.6 * np.log10(np.maximum(d3d, 1e-10))

    def update_shadow(self, delta_distance_list):
        if len(delta_distance_list) == 0:
            self.Shadow = np.random.normal(0, self.shadow_std, self.n_Veh)
            return
        delta_distance = np.asarray(delta_distance_list)
        factor = np.exp(-delta_distance / self.Decorrelation_distance)
        noise = np.random.normal(0, self.shadow_std, self.n_Veh)
        self.Shadow = factor * self.Shadow + np.sqrt(1 - np.exp(-2 * delta_distance / self.Decorrelation_distance)) * noise

    def update_fast_fading(self):
        h = (np.random.normal(size=(self.n_Veh, self.n_RB)) +
             1j * np.random.normal(size=(self.n_Veh, self.n_RB))) / np.sqrt(2)
        self.FastFading = 20 * np.log10(np.abs(h))


# ==================== Vehicle / Environment ====================

class Vehicle:
    def __init__(self, start_position, start_direction, velocity):
        self.position = start_position
        self.direction = start_direction
        self.velocity = velocity
        self.neighbors = []
        self.destinations = []


class Environ:
    def __init__(self, down_lane, up_lane, left_lane, right_lane, width, height):
        self.timestep = 0.01
        self.down_lanes = down_lane
        self.up_lanes = up_lane
        self.left_lanes = left_lane
        self.right_lanes = right_lane
        self.width = width
        self.height = height

        self.vehicles = []
        self.demands = []

        self.V2V_power_dB = 23
        self.V2I_power_dB = 23
        self.V2V_power_dB_List = [23, 10, 5]

        self.sig2_dB = -114
        self.bsAntGain = 8
        self.bsNoiseFigure = 5
        self.vehAntGain = 3
        self.vehNoiseFigure = 9
        self.sig2 = 10 ** (self.sig2_dB / 10)

        self.n_RB = 20
        self.n_Veh = 20

        self.V2Vchannels = V2Vchannels(self.n_Veh, self.n_RB)
        self.V2Ichannels = V2Ichannels(self.n_Veh, self.n_RB)
        self.V2V_Interference_all = np.zeros((self.n_Veh, 3, self.n_RB)) + self.sig2

        self.n_step = 0
        self.Distance = np.zeros((self.n_Veh, self.n_Veh))
        self._slow_channel_valid = False
        self.comm_backend = "numpy"

    # ---------- Vehicle generation ----------

    def add_new_vehicles(self, start_position, start_direction, start_velocity):
        self.vehicles.append(Vehicle(start_position, start_direction, start_velocity))

    def add_new_vehicles_by_number(self, n):
        for _ in range(n):
            ind = np.random.randint(0, len(self.down_lanes))
            self.add_new_vehicles([self.down_lanes[ind], random.randint(0, self.height)], 'd', random.randint(10, 15))
            self.add_new_vehicles([self.up_lanes[ind], random.randint(0, self.height)], 'u', random.randint(10, 15))
            self.add_new_vehicles([random.randint(0, self.width), self.left_lanes[ind]], 'l', random.randint(10, 15))
            self.add_new_vehicles([random.randint(0, self.width), self.right_lanes[ind]], 'r', random.randint(10, 15))
        self.V2V_Shadowing = np.random.normal(0, 3, [len(self.vehicles), len(self.vehicles)])
        self.V2I_Shadowing = np.random.normal(0, 8, len(self.vehicles))
        self.delta_distance = np.asarray([c.velocity for c in self.vehicles])

    # ---------- Mobility ----------

    def renew_positions(self):
        moved = False
        i = 0
        while i < len(self.vehicles):
            v = self.vehicles[i]
            delta_distance = v.velocity * self.timestep
            change_direction = False
            old_position = (float(v.position[0]), float(v.position[1]))

            if v.direction == 'u':
                for y in self.left_lanes:
                    if v.position[1] <= y <= v.position[1] + delta_distance and random.random() < 0.4:
                        v.position = [v.position[0] - (delta_distance - (y - v.position[1])), y]
                        v.direction = 'l'
                        change_direction = True
                        break
                if not change_direction:
                    for y in self.right_lanes:
                        if v.position[1] <= y <= v.position[1] + delta_distance and random.random() < 0.4:
                            v.position = [v.position[0] + (delta_distance + (y - v.position[1])), y]
                            v.direction = 'r'
                            change_direction = True
                            break
                if not change_direction:
                    v.position[1] += delta_distance

            elif v.direction == 'd':
                for y in self.left_lanes:
                    if v.position[1] >= y >= v.position[1] - delta_distance and random.random() < 0.4:
                        v.position = [v.position[0] - (delta_distance - (v.position[1] - y)), y]
                        v.direction = 'l'
                        change_direction = True
                        break
                if not change_direction:
                    for y in self.right_lanes:
                        if v.position[1] >= y >= v.position[1] - delta_distance and random.random() < 0.4:
                            v.position = [v.position[0] + (delta_distance + (v.position[1] - y)), y]
                            v.direction = 'r'
                            change_direction = True
                            break
                if not change_direction:
                    v.position[1] -= delta_distance

            elif v.direction == 'r':
                for x in self.up_lanes:
                    if v.position[0] <= x <= v.position[0] + delta_distance and random.random() < 0.4:
                        v.position = [x, v.position[1] + (delta_distance - (x - v.position[0]))]
                        v.direction = 'u'
                        change_direction = True
                        break
                if not change_direction:
                    for x in self.down_lanes:
                        if v.position[0] <= x <= v.position[0] + delta_distance and random.random() < 0.4:
                            v.position = [x, v.position[1] - (delta_distance - (x - v.position[0]))]
                            v.direction = 'd'
                            change_direction = True
                            break
                if not change_direction:
                    v.position[0] += delta_distance

            else:  # 'l'
                for x in self.up_lanes:
                    if v.position[0] >= x >= v.position[0] - delta_distance and random.random() < 0.4:
                        v.position = [x, v.position[1] + (delta_distance - (v.position[0] - x))]
                        v.direction = 'u'
                        change_direction = True
                        break
                if not change_direction:
                    for x in self.down_lanes:
                        if v.position[0] >= x >= v.position[0] - delta_distance and random.random() < 0.4:
                            v.position = [x, v.position[1] - (delta_distance - (v.position[0] - x))]
                            v.direction = 'd'
                            change_direction = True
                            break
                if not change_direction:
                    v.position[0] -= delta_distance

            # wrap-around strategy
            if (v.position[0] < 0) or (v.position[1] < 0) or (v.position[0] > self.width) or (v.position[1] > self.height):
                if v.direction == 'u':
                    v.direction = 'r'
                    v.position = [v.position[0], self.right_lanes[-1]]
                elif v.direction == 'd':
                    v.direction = 'l'
                    v.position = [v.position[0], self.left_lanes[0]]
                elif v.direction == 'l':
                    v.direction = 'u'
                    v.position = [self.up_lanes[0], v.position[1]]
                else:
                    v.direction = 'd'
                    v.position = [self.down_lanes[-1], v.position[1]]
            if float(v.position[0]) != old_position[0] or float(v.position[1]) != old_position[1]:
                moved = True
            i += 1
        if moved:
            self._slow_channel_valid = False

    # ---------- Neighbor/Destinations ----------

    def renew_neighbor(self):
        N = len(self.vehicles)
        for i in range(N):
            self.vehicles[i].neighbors = []
            self.vehicles[i].actions = []
        if N == 0:
            self.Distance = np.zeros((0, 0))
            return

        z = np.array([[complex(c.position[0], c.position[1]) for c in self.vehicles]])
        Distance = abs(z.T - z)
        self.Distance = Distance

        for i in range(N):
            sort_idx = np.argsort(Distance[:, i])
            others = sort_idx[1:]
            self.vehicles[i].neighbors = list(others[:min(3, len(others))])

            if len(others) == 0:
                self.vehicles[i].destinations = [i, i, i]
                continue

            pool_limit = max(3, min(len(others), int(np.ceil(N / 3))))
            candidate_pool = others[:pool_limit]
            if len(candidate_pool) >= 3:
                dest = np.random.choice(candidate_pool, 3, replace=False)
            else:
                dest = np.random.choice(others, 3, replace=True)
            self.vehicles[i].destinations = dest

    # ---------- Channels ----------

    def renew_channel(self):
        positions = [c.position for c in self.vehicles]
        self.V2Ichannels.update_positions(positions)
        self.V2Vchannels.update_positions(positions)
        self.V2Ichannels.update_pathloss()
        self.V2Vchannels.update_pathloss()
        delta_distance = 0.002 * np.asarray([c.velocity for c in self.vehicles])
        self.V2Ichannels.update_shadow(delta_distance)
        self.V2Vchannels.update_shadow(delta_distance)
        self.V2V_channels_abs = self.V2Vchannels.PathLoss + self.V2Vchannels.Shadow + 50 * np.identity(len(self.vehicles))
        self.V2I_channels_abs = self.V2Ichannels.PathLoss + self.V2Ichannels.Shadow
        self._slow_channel_valid = True

    def _refresh_fastfading_views(self):
        self.V2V_channels_with_fastfading = self.V2V_channels_abs[:, :, None] - self.V2Vchannels.FastFading
        self.V2I_channels_with_fastfading = self.V2I_channels_abs[:, None] - self.V2Ichannels.FastFading

    def renew_channels_fastfading(self, update_fast_fading: bool = True, update_slow_channel: bool = True):
        if update_slow_channel or not getattr(self, "_slow_channel_valid", False):
            self.renew_channel()
        if update_fast_fading:
            self.V2Ichannels.update_fast_fading()
            self.V2Vchannels.update_fast_fading()
        self._refresh_fastfading_views()

    # ---------- Performance / Reward Helpers ----------

    def Compute_Performance_Reward_fast_fading_with_power_asyn(self, actions_power):
        if compute_async_rates is not None and hasattr(self, "topology_state_arrays"):
            destinations = self.topology_state_arrays()["destinations"]
            rate_fn = compute_async_rates
            use_cupy = (
                should_use_cupy(getattr(self, "comm_backend", None), n_links=int(actions_power.shape[0] * actions_power.shape[1]))
                if should_use_cupy is not None else False
            )
            if use_cupy and compute_async_rates_cupy is not None:
                rate_fn = compute_async_rates_cupy
            V2I_Rate, V2V_Rate, V2V_Interference, V2I_Interference = rate_fn(
                actions_power,
                self.activate_links,
                destinations,
                self.V2V_channels_with_fastfading,
                self.V2I_channels_with_fastfading,
                self.V2I_channels_abs,
                power_db_list=self.V2V_power_dB_List,
                v2i_power_db=self.V2I_power_dB,
                veh_ant_gain=self.vehAntGain,
                bs_ant_gain=self.bsAntGain,
                bs_noise_figure=self.bsNoiseFigure,
                veh_noise_figure=self.vehNoiseFigure,
                sig2=self.sig2,
                n_rb=self.n_RB,
                n_vehicles=len(self.vehicles),
            )
            self.V2I_Interference = V2I_Interference
            self.V2V_Interference = V2V_Interference

            self.demand -= V2V_Rate * self.update_time_asyn * 1500
            self.test_time_count -= self.update_time_asyn
            self.individual_time_limit -= self.update_time_asyn
            self.individual_time_interval -= self.update_time_asyn

            new_active = self.individual_time_interval <= 0
            self.activate_links[new_active] = True
            self.individual_time_interval[new_active] = np.random.exponential(
                0.02, self.individual_time_interval[new_active].shape) + self.V2V_limit
            self.individual_time_limit[new_active] = self.V2V_limit
            self.demand[new_active] = self.demand_amount

            early_finish = (self.demand <= 0) & self.activate_links
            unqualified = (self.individual_time_limit <= 0) & self.activate_links
            self.activate_links[early_finish | unqualified] = False
            self.success_transmission += np.sum(early_finish)
            self.failed_transmission += np.sum(unqualified)
            fail_percent = self.failed_transmission / (self.failed_transmission + self.success_transmission + 1e-4)
            return V2I_Rate, fail_percent

        actions = actions_power[:, :, 0]
        power_selection = actions_power[:, :, 1]

        Interference = np.zeros(self.n_RB)
        for i in range(len(self.vehicles)):
            for j in range(actions.shape[1]):
                if not self.activate_links[i, j]:
                    continue
                rb = actions[i, j]
                p_idx = power_selection[i, j]
                Interference[rb] += 10 ** ((self.V2V_power_dB_List[p_idx] -
                                            self.V2I_channels_with_fastfading[i, rb] +
                                            self.vehAntGain + self.bsAntGain - self.bsNoiseFigure) / 10)
        self.V2I_Interference = Interference + self.sig2

        V2V_Interference = np.zeros((len(self.vehicles), 3))
        V2V_Signal = np.zeros((len(self.vehicles), 3))
        actions_mask = actions.copy()
        actions_mask[~self.activate_links] = -1

        for rb in range(self.n_RB):
            idxs = np.argwhere(actions_mask == rb)
            for a in range(len(idxs)):
                tx_i, link_j = idxs[a]
                rx_j = self.vehicles[tx_i].destinations[link_j]
                p_idx = power_selection[tx_i, link_j]
                V2V_Signal[tx_i, link_j] = 10 ** ((self.V2V_power_dB_List[p_idx] -
                                                   self.V2V_channels_with_fastfading[tx_i, rx_j, rb] +
                                                   2 * self.vehAntGain - self.vehNoiseFigure) / 10)
                if rb < self.n_Veh:
                    V2V_Interference[tx_i, link_j] += 10 ** ((self.V2I_power_dB -
                                                              self.V2V_channels_with_fastfading[rb, rx_j, rb] +
                                                              2 * self.vehAntGain - self.vehNoiseFigure) / 10)
                for b in range(a + 1, len(idxs)):
                    tx_k, link_l = idxs[b]
                    rx_l = self.vehicles[tx_k].destinations[link_l]
                    p_idx2 = power_selection[tx_k, link_l]
                    V2V_Interference[tx_i, link_j] += 10 ** ((self.V2V_power_dB_List[p_idx2] -
                                                              self.V2V_channels_with_fastfading[tx_k, rx_j, rb] +
                                                              2 * self.vehAntGain - self.vehNoiseFigure) / 10)
                    V2V_Interference[tx_k, link_l] += 10 ** ((self.V2V_power_dB_List[p_idx] -
                                                              self.V2V_channels_with_fastfading[tx_i, rx_l, rb] +
                                                              2 * self.vehAntGain - self.vehNoiseFigure) / 10)

        self.V2V_Interference = V2V_Interference + self.sig2
        V2V_Rate = np.log2(1 + V2V_Signal / self.V2V_Interference)

        V2I_Signals = (self.V2I_power_dB - self.V2I_channels_abs[0:min(self.n_RB, self.n_Veh)]
                       + self.vehAntGain + self.bsAntGain - self.bsNoiseFigure)
        V2I_Rate = np.log2(1 + 10 ** (V2I_Signals / 10) / self.V2I_Interference[0:min(self.n_RB, self.n_Veh)])

        self.demand -= V2V_Rate * self.update_time_asyn * 1500
        self.test_time_count -= self.update_time_asyn
        self.individual_time_limit -= self.update_time_asyn
        self.individual_time_interval -= self.update_time_asyn

        new_active = self.individual_time_interval <= 0
        self.activate_links[new_active] = True
        self.individual_time_interval[new_active] = np.random.exponential(
            0.02, self.individual_time_interval[new_active].shape) + self.V2V_limit
        self.individual_time_limit[new_active] = self.V2V_limit
        self.demand[new_active] = self.demand_amount

        early_finish = (self.demand <= 0) & self.activate_links
        unqualified = (self.individual_time_limit <= 0) & self.activate_links
        self.activate_links[early_finish | unqualified] = False
        self.success_transmission += np.sum(early_finish)
        self.failed_transmission += np.sum(unqualified)
        fail_percent = self.failed_transmission / (self.failed_transmission + self.success_transmission + 1e-4)
        return V2I_Rate, fail_percent

    # ----------- Training reward (single-link table) -----------

    def Compute_Performance_Reward_Batch(self, actions_power, idx):
        actions = actions_power[:, :, 0]
        power_selection = actions_power[:, :, 1]

        V2I_reward_list = np.zeros((self.n_RB, len(self.V2V_power_dB_List)))
        V2V_reward_list = np.zeros((self.n_RB, len(self.V2V_power_dB_List)))

        tx, link = idx
        if compute_single_link_reward_table is not None and hasattr(self, "topology_state_arrays"):
            rx = int(self.topology_state_arrays()["destinations"][tx, link])
            return compute_single_link_reward_table(
                tx=tx,
                link=link,
                rx=rx,
                demand_value=float(self.demand[tx, link]),
                individual_time_limit_value=float(self.individual_time_limit[tx, link]),
                v2v_channels_with_fastfading=self.V2V_channels_with_fastfading,
                v2i_channels_with_fastfading=self.V2I_channels_with_fastfading,
                v2i_channels_abs=self.V2I_channels_abs,
                power_db_list=self.V2V_power_dB_List,
                v2i_power_db=self.V2I_power_dB,
                veh_ant_gain=self.vehAntGain,
                bs_ant_gain=self.bsAntGain,
                bs_noise_figure=self.bsNoiseFigure,
                veh_noise_figure=self.vehNoiseFigure,
                sig2=self.sig2,
                n_rb=self.n_RB,
                n_vehicles=self.n_Veh,
                v2v_limit=self.V2V_limit,
            )

        rx = self.vehicles[tx].destinations[link]

        for rb in range(self.n_RB):
            for p_i, p_dB in enumerate(self.V2V_power_dB_List):
                signal = 10 ** ((p_dB - self.V2V_channels_with_fastfading[tx, rx, rb] +
                                  2 * self.vehAntGain - self.vehNoiseFigure) / 10)

                interf_v2v = 0.0
                interf_v2v += 10 ** ((self.V2I_power_dB -
                                       self.V2V_channels_with_fastfading[rb % len(self.vehicles), rx, rb] +
                                       2 * self.vehAntGain - self.vehNoiseFigure) / 10)
                v2v_rate = np.log2(1 + signal / (interf_v2v + self.sig2))

                v2i_signal = 10 ** ((self.V2I_power_dB + self.vehAntGain + self.bsAntGain -
                                      self.bsNoiseFigure - self.V2I_channels_abs[min(rb, self.n_Veh - 1)]) / 10)
                extra_i = 10 ** ((p_dB - self.V2I_channels_with_fastfading[tx, rb] +
                                   self.vehAntGain + self.bsAntGain - self.bsNoiseFigure) / 10)
                v2i_rate = np.log2(1 + v2i_signal / (self.sig2 + extra_i))

                V2V_reward_list[rb, p_i] = v2v_rate
                V2I_reward_list[rb, p_i] = v2i_rate

        v2i_scaled = np.tanh(V2I_reward_list / 50.0)
        v2v_scaled = np.tanh(V2V_reward_list / 50.0)
        lam = 0.1
        combo = lam * v2i_scaled + (1 - lam) * v2v_scaled

        if self.demand[tx, link] < 0:
            time_left = self.V2V_limit
        else:
            time_left = self.individual_time_limit[tx, link]
        penalty = (self.V2V_limit - time_left) / self.V2V_limit
        return combo, -penalty, time_left

    # ========== 批量奖励，供 Agent 使用 ==========

    def batch_reward_all(self, actions_power):
        self.n_step += 1
        if self.n_step % 10 == 0:
            self.renew_positions()
            self.renew_channels_fastfading(update_fast_fading=True, update_slow_channel=not getattr(self, "_slow_channel_valid", False))

        actions = actions_power[:, :, 0]
        powers = actions_power[:, :, 1]

        if compute_training_batch_rates is not None and hasattr(self, "topology_state_arrays"):
            rate_fn = compute_training_batch_rates
            use_cupy = (
                should_use_cupy(getattr(self, "comm_backend", None), n_links=int(actions_power.shape[0] * actions_power.shape[1]))
                if should_use_cupy is not None else False
            )
            if use_cupy and compute_training_batch_rates_cupy is not None:
                rate_fn = compute_training_batch_rates_cupy
            V2V_Rate, V2I_Rate, V2I_Interference = rate_fn(
                actions_power,
                self.topology_state_arrays()["destinations"],
                self.V2V_channels_with_fastfading,
                self.V2I_channels_with_fastfading,
                self.V2I_channels_abs,
                power_db_list=self.V2V_power_dB_List,
                v2i_power_db=self.V2I_power_dB,
                veh_ant_gain=self.vehAntGain,
                bs_ant_gain=self.bsAntGain,
                bs_noise_figure=self.bsNoiseFigure,
                veh_noise_figure=self.vehNoiseFigure,
                sig2=self.sig2,
                n_rb=self.n_RB,
                n_vehicles=len(self.vehicles),
            )
            self.V2I_Interference = V2I_Interference
        else:
            # V2I 干扰
            Interference_RB = np.zeros(self.n_RB)
            for i in range(len(self.vehicles)):
                for j in range(3):
                    rb = actions[i, j]
                    if rb < 0 or rb >= self.n_RB:
                        continue
                    p_idx = powers[i, j]
                    Interference_RB[rb] += 10 ** ((self.V2V_power_dB_List[p_idx] -
                                                   self.V2I_channels_with_fastfading[i, rb] +
                                                   self.vehAntGain + self.bsAntGain - self.bsNoiseFigure) / 10)
            self.V2I_Interference = Interference_RB + self.sig2

            # V2V 信号与互扰
            V2V_Signal = np.zeros((len(self.vehicles), 3))
            V2V_Interf = np.zeros((len(self.vehicles), 3))
            for rb in range(self.n_RB):
                idxs = np.argwhere(actions == rb)
                for a in range(len(idxs)):
                    tx_i, link_j = idxs[a]
                    rx_j = self.vehicles[tx_i].destinations[link_j]
                    p_idx = powers[tx_i, link_j]
                    sig = 10 ** ((self.V2V_power_dB_List[p_idx] -
                                  self.V2V_channels_with_fastfading[tx_i, rx_j, rb] +
                                  2 * self.vehAntGain - self.vehNoiseFigure) / 10)
                    V2V_Signal[tx_i, link_j] = sig

                    for b in range(a + 1, len(idxs)):
                        tx_k, link_l = idxs[b]
                        rx_l = self.vehicles[tx_k].destinations[link_l]
                        p_idx2 = powers[tx_k, link_l]
                        interf_ik = 10 ** ((self.V2V_power_dB_List[p_idx2] -
                                            self.V2V_channels_with_fastfading[tx_k, rx_j, rb] +
                                            2 * self.vehAntGain - self.vehNoiseFigure) / 10)
                        interf_ki = 10 ** ((self.V2V_power_dB_List[p_idx] -
                                            self.V2V_channels_with_fastfading[tx_i, rx_l, rb] +
                                            2 * self.vehAntGain - self.vehNoiseFigure) / 10)
                        V2V_Interf[tx_i, link_j] += interf_ik
                        V2V_Interf[tx_k, link_l] += interf_ki

            V2V_Interf = V2V_Interf + self.sig2
            V2V_Rate = np.log2(1 + V2V_Signal / np.maximum(V2V_Interf, 1e-12))

            # V2I 总速率
            V2I_Signals = (self.V2I_power_dB - self.V2I_channels_abs[0:min(self.n_RB, self.n_Veh)]
                           + self.vehAntGain + self.bsAntGain - self.bsNoiseFigure)
            V2I_Rate = np.log2(1 + 10 ** (V2I_Signals / 10) / self.V2I_Interference[0:min(self.n_RB, self.n_Veh)])
        v2i_rate_total = float(np.sum(V2I_Rate))

        # 更新时间与需求
        self.demand -= V2V_Rate * self.update_time_asyn * 1500
        self.test_time_count -= self.update_time_asyn
        self.individual_time_limit -= self.update_time_asyn

        early_finish = (self.demand <= 0)
        unqualified = (self.individual_time_limit <= 0) & (self.demand > 0)
        self.success_transmission += int(np.sum(early_finish))
        self.failed_transmission += int(np.sum(unqualified))
        fail_percent = self.failed_transmission / (self.failed_transmission + self.success_transmission + 1e-6)

        # 基础奖励
        time_left_norm = np.clip(self.individual_time_limit / self.V2V_limit, 0, 1)
        rate_norm = np.tanh(V2V_Rate / 40.0)
        base_reward_matrix = 0.7 * rate_norm + 0.3 * (1 - time_left_norm)

        # 反集中化 + 功率-时间耦合
        from environment_reward_patch import apply_reward_adjustments
        reward_matrix = apply_reward_adjustments(
            base_reward_matrix=base_reward_matrix,
            actions_rb=actions,
            actions_pw=powers,
            individual_time_limit=self.individual_time_limit,
            V2V_limit=self.V2V_limit,
            rb_anti_conc_alpha=getattr(self, "rb_anti_conc_alpha", 0.02),
            rb_hot_threshold=getattr(self, "rb_hot_threshold", 0.18),
            rb_softmask_alpha=getattr(self, "rb_softmask_alpha", 0.25),
            urgency_threshold=getattr(self, "urgency_threshold", 0.30),
            beta_urgency_pos=getattr(self, "beta_urgency_pos", 0.02),
            beta_urgency_neg=getattr(self, "beta_urgency_neg", 0.03),
            n_rb=self.n_RB,
        )

        reset_mask = early_finish | unqualified
        self.demand[reset_mask] = self.demand_amount
        self.individual_time_limit[reset_mask] = self.V2V_limit

        return reward_matrix, v2i_rate_total, float(fail_percent)

    def act_for_training(self, actions, idx):
        rb = actions[idx[0], idx[1], 0]
        pw_idx = actions[idx[0], idx[1], 1]
        reward_table, penalty, _ = self.Compute_Performance_Reward_Batch(actions, idx)
        reward = float(reward_table[rb, pw_idx] + penalty)
        self.renew_positions()
        self.renew_channels_fastfading(update_fast_fading=True, update_slow_channel=not getattr(self, "_slow_channel_valid", False))
        self.Compute_Interference(actions)
        return reward

    def Compute_Interference(self, actions):
        if compute_v2v_interference_db is not None and hasattr(self, "topology_state_arrays"):
            destinations = self.topology_state_arrays()["destinations"]
            interference_fn = compute_v2v_interference_db
            use_cupy = (
                should_use_cupy(getattr(self, "comm_backend", None), n_links=int(actions.shape[0] * actions.shape[1]))
                if should_use_cupy is not None else False
            )
            if use_cupy and compute_v2v_interference_db_cupy is not None:
                interference_fn = compute_v2v_interference_db_cupy
            self.V2V_Interference_all = interference_fn(
                actions,
                destinations,
                self.V2V_channels_with_fastfading,
                power_db_list=self.V2V_power_dB_List,
                veh_ant_gain=self.vehAntGain,
                veh_noise_figure=self.vehNoiseFigure,
                sig2=self.sig2,
                n_rb=self.n_RB,
            )
            return

        V2V_Interference = np.zeros((len(self.vehicles), 3, self.n_RB)) + self.sig2
        channels = actions[:, :, 0].copy()
        powers = actions[:, :, 1].copy()

        for i in range(len(self.vehicles)):
            for j in range(3):
                rb = channels[i, j]
                if rb < 0 or rb >= self.n_RB:
                    continue
                p_sel = self.V2V_power_dB_List[powers[i, j]]
                rx = self.vehicles[i].destinations[j]
                for k in range(len(self.vehicles)):
                    for m in range(3):
                        if k == i and m == j:
                            continue
                        rb2 = channels[k, m]
                        if rb2 != rb:
                            continue
                        rx2 = self.vehicles[k].destinations[m]
                        interf = 10 ** ((p_sel - self.V2V_channels_with_fastfading[i, rx2, rb] +
                                          2 * self.vehAntGain - self.vehNoiseFigure) / 10)
                        V2V_Interference[k, m, rb] += interf

        self.V2V_Interference_all = 10 * np.log10(V2V_Interference)

    def act_asyn(self, actions):
        self.n_step += 1
        if self.n_step % 10 == 0:
            self.renew_positions()
            self.renew_channels_fastfading(update_fast_fading=True, update_slow_channel=not getattr(self, "_slow_channel_valid", False))
        v2i, fail = self.Compute_Performance_Reward_fast_fading_with_power_asyn(actions)
        self.Compute_Interference(actions)
        return v2i, fail

    def act(self, actions):
        self.n_step += 1
        v2i, fail = self.Compute_Performance_Reward_fast_fading_with_power_asyn(actions)
        self.renew_positions()
        self.renew_channels_fastfading(update_fast_fading=True, update_slow_channel=not getattr(self, "_slow_channel_valid", False))
        self.Compute_Interference(actions)
        return v2i, fail

    def Compute_Performance_Reward_fast_fading_with_power(self, actions_power):
        return self.Compute_Performance_Reward_fast_fading_with_power_asyn(actions_power)

    def new_random_game(self, n_Veh=0):
        self.n_step = 0
        self.vehicles = []
        if n_Veh > 0:
            self.n_Veh = n_Veh
        self.Distance = np.zeros((self.n_Veh, self.n_Veh))
        self.add_new_vehicles_by_number(int(self.n_Veh / 4))

        self.V2Vchannels = V2Vchannels(self.n_Veh, self.n_RB)
        self.V2Ichannels = V2Ichannels(self.n_Veh, self.n_RB)
        self._slow_channel_valid = False
        self.renew_channels_fastfading(update_fast_fading=True, update_slow_channel=True)
        self.renew_neighbor()

        self.V2V_Interference_all = np.zeros((self.n_Veh, 3, self.n_RB)) + self.sig2

        self.demand_amount = 30
        self.d
