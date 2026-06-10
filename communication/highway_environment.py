# -*- coding: utf-8 -*-
"""
HighwayTopoEnv (dual 4-lane per direction) with V2I modes:
- Geometry: two directions, each with N lanes (default 4), vehicles placed at equal longitudinal spacing
- Motion: up (+Y) from bottom, down (-Y) from top; stop at opposite end (no wrap)
- Topology per direction: STAR or TREE; leader can be foremost, lane-pinned, and dynamic
- V2I modes:
  * single: single eNB/BS using Environment.V2Ichannels formula (BS_position configurable)
  * rsu: multi-RSU layout with nearest attachment and handover
- Visualization: fixed axes (Y in [0, height]), draw lanes, vehicles, neighbors, BS/RSUs, optional V2I dashed lines
"""

from typing import List, Optional, Tuple
import numpy as np
from Environment import Environ, Vehicle
from tensorized_step import (
    associate_v2i_rsu,
    associate_v2i_single,
    pairwise_distance_matrix,
    step_longitudinal_positions,
)

try:
    from formation.traffic_population import (
        SpawnConfig,
        TrafficCompositionConfig,
        VehicleProfile,
        even_split,
        generate_random_population,
    )
    from formation.mpr_utils import exact_cav_count
except Exception:  # pragma: no cover
    SpawnConfig = None  # type: ignore
    TrafficCompositionConfig = None  # type: ignore
    VehicleProfile = None  # type: ignore
    even_split = None  # type: ignore
    generate_random_population = None  # type: ignore
    exact_cav_count = None  # type: ignore


class HighwayTopoEnv(Environ):
    def __init__(self,
                 n_up: int = 10,
                 n_down: int = 20,
                 lanes_per_dir: int = 4,
                 lane_width: float = 3.5,
                 median_gap_factor: float = 1.2,
                 spacing: float = 20.0,
                 base_y: float = 50.0,
                 width: float = 180.0,
                 height: float = 1400.0,
                 topology_type: str = 'star',      # 'star' or 'tree'
                 jitter_std: float = 0.0,
                 move_speed: float = 0.0,          # >0: unified step (m/step); else per-vehicle velocity * timestep
                 mpr_cav: float = 1.0,
                 random_spawn: bool = False,
                 spawn_y_min: float = 0.0,
                 spawn_y_max: float = 260.0,
                 lane_density_jitter: float = 0.35,
                 min_spawn_gap: float = 11.0,
                 hv_conservative_ratio: float = 0.30,
                 hv_aggressive_ratio: float = 0.20,
                 # Leader policy
                 leader_at_front: bool = True,
                 leader_lane_up: Optional[int] = None,
                 leader_lane_down: Optional[int] = None,
                 leader_dynamic: bool = False,
                 # V2I mode
                 v2i_mode: str = "single",         # 'single' | 'rsu'
                 bs_single_position: Optional[Tuple[float, float]] = None,  # only for 'single' mode
                 # RSU / V2I (for 'rsu' mode)
                 bs_layout: str = "median",        # 'median' | 'dual-roadside'
                 bs_spacing: float = 250.0,        # meters; <=0 disables RSU placement
                 bs_min_stay_steps: int = 5,       # min steps to stay before handover
                 bs_handover_hyst_m: float = 15.0, # meters of hysteresis to avoid ping-pong
                 seed: int = 123,
                 use_tensorized_position_step: bool = False):
        np.random.seed(seed)
        self.rng = np.random.default_rng(seed)
        assert topology_type in ('star', 'tree')
        assert v2i_mode in ('single', 'rsu')
        assert bs_layout in ('median', 'dual-roadside')
        self.n_up = int(n_up)
        self.n_down = int(n_down)
        self.lanes_per_dir = int(lanes_per_dir)
        self.lane_width = float(lane_width)
        self.median_gap = float(median_gap_factor * lane_width)
        self.spacing = float(spacing)
        self.base_y = float(base_y)
        self.topology_type = topology_type
        self.jitter_std = float(jitter_std)
        self.move_speed = float(move_speed)
        self.mpr_cav = float(np.clip(mpr_cav, 0.0, 1.0))
        self.random_spawn = bool(random_spawn)
        self.spawn_y_min = float(spawn_y_min)
        self.spawn_y_max = float(spawn_y_max)
        self.lane_density_jitter = float(lane_density_jitter)
        self.min_spawn_gap = float(min_spawn_gap)
        self.hv_conservative_ratio = float(hv_conservative_ratio)
        self.hv_aggressive_ratio = float(hv_aggressive_ratio)
        self.leader_at_front = bool(leader_at_front)
        self.leader_lane_up = leader_lane_up
        self.leader_lane_down = leader_lane_down
        self.leader_dynamic = bool(leader_dynamic)
        self.seed = seed
        self.width = float(width)
        self.height = float(height)
        # V2I mode/config
        self.v2i_mode = v2i_mode
        self.bs_single_position = bs_single_position  # may be None -> center
        # RSU config (used only in 'rsu' mode)
        self.bs_layout = bs_layout
        self.bs_spacing = float(bs_spacing)
        self.bs_min_stay_steps = int(bs_min_stay_steps)
        self.bs_handover_hyst_m = float(bs_handover_hyst_m)
        self.use_tensorized_position_step = bool(use_tensorized_position_step)

        # Lane x positions: up lanes on the left of center, down lanes on the right
        center_x = width / 2.0
        up_lanes = [center_x - self.median_gap / 2.0 - (k + 0.5) * self.lane_width
                    for k in range(self.lanes_per_dir)]
        down_lanes = [center_x + self.median_gap / 2.0 + (k + 0.5) * self.lane_width
                      for k in range(self.lanes_per_dir)]
        # Unused lateral placeholders (kept for Environ compatibility)
        left_lanes = [0.0]
        right_lanes = [0.0]

        super().__init__(down_lanes, up_lanes, left_lanes, right_lanes, width, height)
        self.topology_epoch = 0

        self.true_up_lanes = up_lanes
        self.true_down_lanes = down_lanes

        if TrafficCompositionConfig is not None and SpawnConfig is not None:
            exact_count = -1
            if exact_cav_count is not None:
                exact_count = int(exact_cav_count(self.n_up + self.n_down, self.mpr_cav))
            self.population_config = TrafficCompositionConfig(
                mpr_cav=self.mpr_cav,
                exact_cav_count=exact_count,
                hv_conservative_ratio=self.hv_conservative_ratio,
                hv_aggressive_ratio=self.hv_aggressive_ratio,
            )
            self.spawn_config = SpawnConfig(
                random_spawn=self.random_spawn,
                spawn_y_min=self.spawn_y_min,
                spawn_y_max=self.spawn_y_max,
                spacing=self.spacing,
                lane_density_jitter=self.lane_density_jitter,
                min_spawn_gap=self.min_spawn_gap,
            )
        else:
            self.population_config = None
            self.spawn_config = None

        self.cav_indices = np.zeros(0, dtype=int)
        self.hv_indices = np.zeros(0, dtype=int)
        self.new_random_game()

    # ---------------- Placement ----------------
    def _even_split(self, total: int, k: int) -> List[int]:
        if even_split is not None:
            return list(even_split(total, k))
        q, r = divmod(total, k)
        return [q + (1 if i < r else 0) for i in range(k)]

    def _apply_vehicle_profile(self, veh: Vehicle, profile: VehicleProfile, lane_idx: int) -> None:
        veh.veh_type = str(profile.veh_type)
        veh.is_cav = bool(profile.is_cav)
        veh.driver_style = str(profile.driver_style)
        veh.desired_speed = float(profile.desired_speed)
        veh.desired_headway = float(profile.desired_headway)
        veh.desired_standstill_gap = float(profile.desired_standstill_gap)
        veh.comfortable_brake = float(profile.comfortable_brake)
        veh.politeness = float(profile.politeness)
        veh.vehicle_length = float(profile.vehicle_length)
        veh.accel_limit = float(profile.accel_limit)
        veh.lane_change_cooldown_s = float(profile.lane_change_cooldown_s)
        veh.mobil_accel_threshold = float(profile.mobil_accel_threshold)
        veh.mobil_safe_brake = float(profile.mobil_safe_brake)
        veh.mobil_right_bias = float(profile.mobil_right_bias)
        veh.perception_head_m = float(profile.perception_head_m)
        veh.perception_tail_m = float(profile.perception_tail_m)
        veh.last_lane_change_step = -10**9
        veh.home_lane_idx = int(lane_idx)

    def _build_positions_dual_4lane(self):
        self.vehicles = []
        self._lane_idx = []  # per-vehicle lane index (0..lanes-1), used to select leader
        generated = []
        if generate_random_population is not None and self.population_config is not None and self.spawn_config is not None:
            generated = generate_random_population(
                n_up=self.n_up,
                n_down=self.n_down,
                lanes_per_dir=self.lanes_per_dir,
                lane_positions_up=self.true_up_lanes,
                lane_positions_down=self.true_down_lanes,
                height=self.height,
                base_y=self.base_y,
                composition=self.population_config,
                spawn=self.spawn_config,
                rng=self.rng,
            )
        if generated:
            generated.sort(key=lambda rec: (0 if rec["direction"] == "u" else 1, int(rec["lane_idx"]), float(rec["y"])))
            for rec in generated:
                profile = rec["profile"]
                veh = Vehicle([float(rec["x"]), float(rec["y"])], str(rec["direction"]), float(profile.desired_speed))
                self._apply_vehicle_profile(veh, profile, int(rec["lane_idx"]))
                self.vehicles.append(veh)
                self._lane_idx.append(int(rec["lane_idx"]))
        else:
            dist_up = self._even_split(self.n_up, self.lanes_per_dir)
            dist_down = self._even_split(self.n_down, self.lanes_per_dir)

            for lane_idx, cnt in enumerate(dist_up):
                x = self.true_up_lanes[lane_idx]
                for t in range(cnt):
                    y = self.base_y + t * self.spacing
                    v = max(0.0, 22.0 + np.random.normal(0.0, 2.5))
                    veh = Vehicle([x, y], 'u', velocity=v)
                    veh.veh_type = "cav"
                    veh.is_cav = True
                    veh.driver_style = "cooperative"
                    veh.desired_speed = float(v)
                    veh.desired_headway = 1.0
                    veh.comfortable_brake = 2.8
                    veh.politeness = 0.30
                    veh.vehicle_length = 4.8
                    veh.accel_limit = 2.4
                    veh.lane_change_cooldown_s = 5.0
                    veh.last_lane_change_step = -10**9
                    veh.home_lane_idx = int(lane_idx)
                    self.vehicles.append(veh)
                    self._lane_idx.append(lane_idx)

            for lane_idx, cnt in enumerate(dist_down):
                x = self.true_down_lanes[lane_idx]
                for t in range(cnt):
                    y = (self.height - self.base_y) - t * self.spacing
                    v = max(0.0, 22.0 + np.random.normal(0.0, 2.5))
                    veh = Vehicle([x, y], 'd', velocity=v)
                    veh.veh_type = "cav"
                    veh.is_cav = True
                    veh.driver_style = "cooperative"
                    veh.desired_speed = float(v)
                    veh.desired_headway = 1.0
                    veh.comfortable_brake = 2.8
                    veh.politeness = 0.30
                    veh.vehicle_length = 4.8
                    veh.accel_limit = 2.4
                    veh.lane_change_cooldown_s = 5.0
                    veh.last_lane_change_step = -10**9
                    veh.home_lane_idx = int(lane_idx)
                    self.vehicles.append(veh)
                    self._lane_idx.append(lane_idx)

        self.n_Veh = len(self.vehicles)
        self._group_dir = [str(getattr(veh, "direction", "u")) for veh in self.vehicles]
        self.cav_indices = np.asarray([idx for idx, veh in enumerate(self.vehicles) if bool(getattr(veh, "is_cav", False))], dtype=int)
        self.hv_indices = np.asarray([idx for idx, veh in enumerate(self.vehicles) if not bool(getattr(veh, "is_cav", False))], dtype=int)

    # ---------------- BS/RSU & V2I ----------------
    def _place_base_stations(self):
        """Generate base-station positions for visualization and association.

        - single mode: one BS at bs_single_position (or center if None)
        - rsu mode: a series of RSUs along the road as before
        """
        self.bs_positions: List[Tuple[float, float]] = []
        if self.v2i_mode == "single":
            pos = self.bs_single_position or (self.width / 2.0, self.height / 2.0)
            self.bs_positions = [pos]
            return

        # rsu mode
        if self.bs_spacing <= 0:
            return
        center_x = self.width / 2.0
        x_left = min(self.true_up_lanes) - self.lane_width
        x_right = max(self.true_down_lanes) + self.lane_width

        # y grid with half-spacing offset
        y_list: List[float] = []
        y = self.base_y + 0.5 * self.bs_spacing
        while y <= self.height - self.base_y:
            y_list.append(y)
            y += self.bs_spacing

        if self.bs_layout == "median":
            self.bs_positions = [(center_x, yy) for yy in y_list]
        else:  # dual-roadside
            for i, yy in enumerate(y_list):
                self.bs_positions.append((x_left if (i % 2 == 0) else x_right, yy))

    def _assoc_v2i(self, initial: bool = False):
        """
        Associate each vehicle to a serving BS/RSU.
        - single: everyone attaches to index 0; v2i_dist_m = distance to that BS
        - rsu: nearest-RSU with hysteresis + min-stay to avoid ping-pong
        """
        if self.n_Veh == 0:
            self.v2i_serving_idx = np.zeros((0,), dtype=int)
            self.v2i_stay_steps = np.zeros((0,), dtype=int)
            self.v2i_dist_m = np.zeros((0,), dtype=float)
            return

        if self.v2i_mode == "single":
            if not self.bs_positions:
                self.bs_positions = [self.bs_single_position or (self.width / 2.0, self.height / 2.0)]
            state = self.vehicle_state_arrays()
            serving, stay, dist = associate_v2i_single(
                state["positions_xy"],
                np.asarray(self.bs_positions[0], dtype=np.float32),
                stay_steps=getattr(self, "v2i_stay_steps", None),
                initial=initial,
            )
            self.v2i_serving_idx = serving
            self.v2i_stay_steps = stay
            self.v2i_dist_m = dist.astype(float, copy=False)
            return

        # rsu mode
        n_bs = len(self.bs_positions)
        if n_bs == 0:
            self.v2i_serving_idx = np.full((self.n_Veh,), -1, dtype=int)
            self.v2i_stay_steps = np.zeros((self.n_Veh,), dtype=int)
            self.v2i_dist_m = np.zeros((self.n_Veh,), dtype=float)
            return

        state = self.vehicle_state_arrays()
        serving, stay, dist = associate_v2i_rsu(
            state["positions_xy"],
            np.asarray(self.bs_positions, dtype=np.float32),
            current_serving_idx=None if initial else getattr(self, "v2i_serving_idx", None),
            current_dist_m=None if initial else getattr(self, "v2i_dist_m", None),
            stay_steps=None if initial else getattr(self, "v2i_stay_steps", None),
            hysteresis_m=self.bs_handover_hyst_m,
            min_stay_steps=self.bs_min_stay_steps,
            initial=initial,
        )
        self.v2i_serving_idx = serving
        self.v2i_stay_steps = stay
        self.v2i_dist_m = dist.astype(float, copy=False)

    # ---------------- Leader selection ----------------
    def _default_center_lanes(self):
        # Even lanes: up uses "middle-left", down uses "middle-right"
        if self.lanes_per_dir % 2 == 0:
            up_idx = self.lanes_per_dir // 2 - 1
            down_idx = self.lanes_per_dir // 2
        else:
            up_idx = down_idx = self.lanes_per_dir // 2
        return up_idx, down_idx

    def _pick_leaders_by_policy(self):
        def clamp_lane(i: int) -> int:
            return int(max(0, min(self.lanes_per_dir - 1, i)))

        up_center, down_center = self._default_center_lanes()
        up_lane = clamp_lane(self.leader_lane_up if self.leader_lane_up is not None else up_center)
        down_lane = clamp_lane(self.leader_lane_down if self.leader_lane_down is not None else down_center)

        up_idxs = [i for i, d in enumerate(self._group_dir) if d == 'u']
        down_idxs = [i for i, d in enumerate(self._group_dir) if d == 'd']

        def pick_from_lane(cands: List[int], lane_idx: int, direction: str) -> Optional[int]:
            lane_cands = [i for i in cands if self._lane_idx[i] == lane_idx]
            lane_cands_cav = [i for i in lane_cands if bool(getattr(self.vehicles[i], "is_cav", False))]
            pool = lane_cands_cav or lane_cands or [i for i in cands if bool(getattr(self.vehicles[i], "is_cav", False))] or cands
            key = (lambda i: self.vehicles[i].position[1])
            if not cands:
                return None
            if direction == 'u':  # foremost = max y
                return max(pool, key=key, default=max(cands, key=key))
            else:                 # 'd' foremost = min y
                return min(pool, key=key, default=min(cands, key=key))

        self.leader_idx_up = pick_from_lane(up_idxs, up_lane, 'u') if up_idxs else None
        self.leader_idx_down = pick_from_lane(down_idxs, down_lane, 'd') if down_idxs else None

    def _refresh_topology_tensor_cache(self) -> None:
        n_vehicles = int(self.n_Veh)
        destinations = np.full((n_vehicles, 3), -1, dtype=np.int32)
        max_neighbor_count = max((len(getattr(v, "neighbors", [])) for v in self.vehicles), default=0)
        neighbors = np.full((n_vehicles, max_neighbor_count), -1, dtype=np.int32)
        degrees = np.zeros((n_vehicles,), dtype=np.int32)
        struct_features = np.asarray(getattr(self, "struct_features_per_vehicle", []), dtype=np.float32).reshape(-1, 2)
        if struct_features.shape[0] != n_vehicles:
            struct_features = np.zeros((n_vehicles, 2), dtype=np.float32)

        for idx, veh in enumerate(self.vehicles):
            dest_list = [int(x) for x in getattr(veh, "destinations", [])[:3]]
            if dest_list:
                destinations[idx, : len(dest_list)] = np.asarray(dest_list, dtype=np.int32)
            neigh_list = [int(x) for x in getattr(veh, "neighbors", [])]
            if neigh_list:
                neighbors[idx, : len(neigh_list)] = np.asarray(neigh_list, dtype=np.int32)
                degrees[idx] = len(neigh_list)

        leader_up = -1 if self.leader_idx_up is None else int(self.leader_idx_up)
        leader_down = -1 if self.leader_idx_down is None else int(self.leader_idx_down)
        leader_flags = np.zeros((n_vehicles, 2), dtype=np.bool_)
        if 0 <= leader_up < n_vehicles:
            leader_flags[leader_up, 0] = True
        if 0 <= leader_down < n_vehicles:
            leader_flags[leader_down, 1] = True

        self.topology_arrays = {
            "destinations": destinations,
            "neighbors": neighbors,
            "degrees": degrees,
            "depth": np.asarray(getattr(self, "depth", np.zeros((n_vehicles,), dtype=int)), dtype=np.int32).copy(),
            "struct_features": struct_features,
            "leader_indices": np.asarray([leader_up, leader_down], dtype=np.int32),
            "leader_flags": leader_flags,
            "topology_epoch": np.asarray([int(getattr(self, "topology_epoch", 0))], dtype=np.int32),
        }

    def topology_state_arrays(self) -> dict[str, np.ndarray]:
        if not hasattr(self, "topology_arrays"):
            self._refresh_topology_tensor_cache()
        return {key: value.copy() for key, value in self.topology_arrays.items()}

    def v2v_adjacency_matrix(self) -> np.ndarray:
        topo = self.topology_state_arrays()
        destinations = np.asarray(topo["destinations"], dtype=np.int32)
        n_vehicles = int(self.n_Veh)
        adjacency = np.zeros((n_vehicles, n_vehicles), dtype=np.float32)
        if n_vehicles == 0 or destinations.size == 0:
            return adjacency
        src = np.repeat(np.arange(n_vehicles, dtype=np.int32), destinations.shape[1])
        dst = destinations.reshape(-1)
        valid = (dst >= 0) & (dst < n_vehicles)
        adjacency[src[valid], dst[valid]] = 1.0
        return adjacency

    # ---------------- Topology ----------------
    def _order_by_front(self, indices: List[int], direction: str) -> List[int]:
        if direction == 'u':
            return sorted(indices, key=lambda i: self.vehicles[i].position[1], reverse=True)
        else:
            return sorted(indices, key=lambda i: self.vehicles[i].position[1])

    def _build_topology_two_clusters(self):
        for v in self.vehicles:
            v.destinations = []
            v.neighbors = []
        self.depth = np.zeros(self.n_Veh, dtype=int)

        up_idxs = [i for i, d in enumerate(self._group_dir) if d == 'u']
        down_idxs = [i for i, d in enumerate(self._group_dir) if d == 'd']

        if up_idxs:
            self._build_cluster(up_idxs, direction='u', fixed_leader=self.leader_idx_up)
        if down_idxs:
            self._build_cluster(down_idxs, direction='d', fixed_leader=self.leader_idx_down)

        max_depth = int(max(1, self.depth.max()))
        self.struct_features_per_vehicle = []
        deg = [len(self.vehicles[i].neighbors) for i in range(self.n_Veh)]
        up_deg_max = max([deg[i] for i in up_idxs]) if up_idxs else 1
        down_deg_max = max([deg[i] for i in down_idxs]) if down_idxs else 1
        for i in range(self.n_Veh):
            depth_norm = float(self.depth[i]) / max_depth
            if i in up_idxs:
                is_hub = 1.0 if deg[i] >= up_deg_max else 0.0
            elif i in down_idxs:
                is_hub = 1.0 if deg[i] >= down_deg_max else 0.0
            else:
                is_hub = 0.0
            self.struct_features_per_vehicle.append((depth_norm, is_hub))
        self.topology_epoch = int(getattr(self, "topology_epoch", 0)) + 1
        self._refresh_topology_tensor_cache()

    def _build_cluster(self, indices: List[int], direction: str, fixed_leader: Optional[int] = None):
        order = self._order_by_front(indices, direction)
        if fixed_leader is not None and fixed_leader in order:
            order.remove(fixed_leader)
            order = [fixed_leader] + order  # force leader first
        elif self.leader_at_front:
            pass  # already foremost

        if self.topology_type == 'star':
            hub = order[0]
            leaves = order[1:4] if len(order) > 1 else [hub, hub, hub]
            while len(leaves) < 3:
                leaves.append(leaves[-1])
            self.vehicles[hub].destinations = leaves  # leader broadcasts to rear
            for i in order[1:]:
                self.vehicles[i].destinations = [hub, hub, hub]  # allow uplink to leader
                self.vehicles[i].neighbors.append(hub)
                self.vehicles[hub].neighbors.append(i)
                self.depth[i] = 1
            self.depth[hub] = 0
            self.vehicles[hub].neighbors = sorted(list(set(self.vehicles[hub].neighbors)))
        else:
            L = len(order)
            for local_i, gidx in enumerate(order):
                left_local = 2 * local_i + 1
                right_local = 2 * local_i + 2
                parent_local = (local_i - 1) // 2 if local_i > 0 else 0
                if local_i == 0:
                    dest = []
                    if left_local < L:
                        dest.append(order[left_local])
                    if right_local < L:
                        dest.append(order[right_local])
                    if left_local < L:
                        dest.append(order[left_local])
                    while len(dest) < 3 and dest:
                        dest.append(dest[-1])
                    if not dest:
                        dest = [gidx, gidx, gidx]
                else:
                    children = []
                    if left_local < L:
                        children.append(order[left_local])
                    if right_local < L:
                        children.append(order[right_local])
                    if children:
                        while len(children) < 2:
                            children.append(children[0])
                        dest = [children[0], children[1], order[parent_local]]
                    else:
                        dest = [order[parent_local]] * 3
                self.vehicles[gidx].destinations = dest
                if left_local < L:
                    self.vehicles[gidx].neighbors.append(order[left_local])
                    self.vehicles[order[left_local]].neighbors.append(gidx)
                if right_local < L:
                    self.vehicles[gidx].neighbors.append(order[right_local])
                    self.vehicles[order[right_local]].neighbors.append(gidx)
                d = 0
                p = local_i
                while p > 0:
                    p = (p - 1) // 2
                    d += 1
                self.depth[gidx] = d

    # ---------------- Session and evolution ----------------
    def _init_session(self):
        self.Distance = np.zeros((self.n_Veh, self.n_Veh))
        self.V2Vchannels = self.V2Vchannels.__class__(self.n_Veh, self.n_RB)
        self.V2Ichannels = self.V2Ichannels.__class__(self.n_Veh, self.n_RB)

        # IMPORTANT: set BS_position for single-BS mode BEFORE channel renewal,
        # so V2I pathloss uses the correct coordinates.
        if self.v2i_mode == "single":
            pos = self.bs_single_position or (self.width / 2.0, self.height / 2.0)
            # Environment.V2Ichannels expects a list [x, y]
            self.V2Ichannels.BS_position = [float(pos[0]), float(pos[1])]

        self.renew_channels_fastfading(update_fast_fading=True, update_slow_channel=True)
        self._update_distance()

        self.demand_amount = 30
        self.demand = self.demand_amount * np.ones((self.n_Veh, 3))
        self.test_time_count = 10
        self.V2V_limit = 0.1
        self.individual_time_limit = self.V2V_limit * np.ones((self.n_Veh, 3))
        self.individual_time_interval = np.random.exponential(0.05, (self.n_Veh, 3))
        self.UnsuccessfulLink = np.zeros((self.n_Veh, 3))
        self.success_transmission = 0
        self.failed_transmission = 0
        self.update_time_train = 0.01
        self.update_time_test = 0.002
        self.update_time_asyn = 0.0002
        self.activate_links = np.zeros((self.n_Veh, 3), dtype='bool')
        self.V2V_Interference_all = np.zeros((self.n_Veh, 3, self.n_RB)) + self.sig2
        self.n_step = 0

    def _update_distance(self):
        state = self.vehicle_state_arrays()
        self.Distance = pairwise_distance_matrix(state["positions_xy"])

    def vehicle_state_arrays(self) -> dict[str, np.ndarray]:
        positions_xy = np.asarray([veh.position for veh in self.vehicles], dtype=np.float32).reshape(-1, 2)
        velocities = np.asarray([float(getattr(veh, "velocity", 0.0)) for veh in self.vehicles], dtype=np.float32)
        lane_indices = np.asarray(getattr(self, "_lane_idx", np.zeros(len(self.vehicles), dtype=int)), dtype=np.int32).copy()
        direction_sign = np.asarray(
            [1.0 if str(getattr(veh, "direction", "u")).lower() == "u" else -1.0 for veh in self.vehicles],
            dtype=np.float32,
        )
        heading_error = np.asarray([float(getattr(veh, "heading_error", 0.0)) for veh in self.vehicles], dtype=np.float32)
        yaw = np.asarray([float(getattr(veh, "yaw", 0.0)) for veh in self.vehicles], dtype=np.float32)
        return {
            "positions_xy": positions_xy,
            "velocities": velocities,
            "lane_indices": lane_indices,
            "direction_sign": direction_sign,
            "heading_error": heading_error,
            "yaw": yaw,
        }

    def apply_vehicle_state_arrays(self, arrays: dict[str, np.ndarray]) -> None:
        positions_xy = np.asarray(arrays["positions_xy"], dtype=np.float32)
        n_vehicles = len(self.vehicles)
        if positions_xy.shape != (n_vehicles, 2):
            raise ValueError(f"positions_xy shape mismatch: expected {(n_vehicles, 2)}, got {positions_xy.shape}")

        velocities = np.asarray(arrays.get("velocities", np.zeros(n_vehicles)), dtype=np.float32).reshape(-1)
        if velocities.shape[0] != n_vehicles:
            raise ValueError("velocities length mismatch")

        lane_indices = np.asarray(arrays.get("lane_indices", getattr(self, "_lane_idx", np.zeros(n_vehicles))), dtype=np.int32).reshape(-1)
        if lane_indices.shape[0] != n_vehicles:
            raise ValueError("lane_indices length mismatch")

        heading_error = arrays.get("heading_error")
        if heading_error is not None:
            heading_error = np.asarray(heading_error, dtype=np.float32).reshape(-1)
            if heading_error.shape[0] != n_vehicles:
                raise ValueError("heading_error length mismatch")

        yaw = arrays.get("yaw")
        if yaw is not None:
            yaw = np.asarray(yaw, dtype=np.float32).reshape(-1)
            if yaw.shape[0] != n_vehicles:
                raise ValueError("yaw length mismatch")

        for idx, veh in enumerate(self.vehicles):
            veh.position[0] = float(positions_xy[idx, 0])
            veh.position[1] = float(positions_xy[idx, 1])
            veh.velocity = float(velocities[idx])
            if heading_error is not None:
                veh.heading_error = float(heading_error[idx])
            if yaw is not None:
                veh.yaw = float(yaw[idx])
        self._lane_idx = lane_indices.astype(int).tolist()

    def set_tensorized_position_step(self, enabled: bool = True) -> None:
        self.use_tensorized_position_step = bool(enabled)

    def renew_positions_tensorized(self) -> bool:
        state = self.vehicle_state_arrays()
        result = step_longitudinal_positions(
            state["positions_xy"],
            state["velocities"],
            state["direction_sign"],
            move_speed=self.move_speed,
            timestep=self.timestep,
            base_y=self.base_y,
            height=self.height,
            jitter_std=self.jitter_std,
            rng=self.rng,
        )
        state["positions_xy"] = result.positions_xy
        self.apply_vehicle_state_arrays(state)
        return bool(np.any(result.moved_mask))

    def _finalize_motion_update(self, moved: bool) -> None:
        if not moved:
            return
        self._slow_channel_valid = False
        self._update_distance()
        if self.leader_dynamic:
            old_up, old_dn = self.leader_idx_up, self.leader_idx_down
            self._pick_leaders_by_policy()
            if self.leader_idx_up != old_up or self.leader_idx_down != old_dn:
                self._build_topology_two_clusters()
        self._assoc_v2i(initial=False)

    def renew_positions(self):
        """
        Up: +Y, Down: -Y; stop at boundaries (no wrap).
        Step priority: move_speed (m/step) > vehicle.velocity * timestep.
        If leader_dynamic=True and the foremost vehicle changes, rebuild topology to set new leader.
        Also updates V2I association (handover rules apply for 'rsu'; single just refreshes distances).
        """
        if self.use_tensorized_position_step:
            moved = self.renew_positions_tensorized()
            self._finalize_motion_update(moved)
            return

        moved = False
        for v in self.vehicles:
            dy = self.move_speed if self.move_speed > 0 else float(getattr(v, 'velocity', 0.0)) * float(self.timestep)
            if dy <= 0.0 and self.jitter_std <= 0:
                continue

            y_old = v.position[1]
            if v.direction == 'u':
                v.position[1] = min(y_old + dy, self.height - self.base_y)
            elif v.direction == 'd':
                v.position[1] = max(y_old - dy, self.base_y)
            else:
                v.position[1] = min(y_old + dy, self.height - self.base_y)

            if self.jitter_std > 0:
                v.position[1] += np.random.normal(0.0, self.jitter_std)
                v.position[1] = float(np.clip(v.position[1], self.base_y, self.height - self.base_y))

            if v.position[1] != y_old:
                moved = True
        self._finalize_motion_update(moved)

    def post_position_update(self):
        """Refresh topology- and channel-related state after external motion updates."""
        self._finalize_motion_update(True)

    def new_random_game(self, n_Veh: int = 0):
        if n_Veh > 0:
            self.n_up = int(max(1, n_Veh // 2))
            self.n_down = int(max(1, n_Veh - self.n_up))
        self._build_positions_dual_4lane()
        self._place_base_stations()
        self._assoc_v2i(initial=True)
        self._pick_leaders_by_policy()
        self._build_topology_two_clusters()
        self._init_session()

    def renew_neighbor(self):
        return  # fixed topology

    # ---------------- Topology switching ----------------
    def set_topology(self, topology_type: str):
        if topology_type not in ('star', 'tree'):
            raise ValueError("topology_type must be 'star' or 'tree'")
        self.topology_type = topology_type
        self._build_topology_two_clusters()
        self._init_session()
        print(f"[HighwayTopoEnv] Topology switched to: {self.topology_type}")

    def toggle_topology(self):
        self.set_topology('tree' if self.topology_type == 'star' else 'star')

    # ---------------- Visualization (fixed axes) ----------------
    def visualize(self,
                  save_path: str = 'highway_topology.png',
                  show_destinations: bool = False,
                  annotate_power: bool = False,
                  agent=None,
                  figsize=(8, 10),
                  dpi: int = 220,
                  show_v2i: bool = False):
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        import matplotlib.patches as mpatches

        fig, ax = plt.subplots(figsize=figsize)
        ax.set_facecolor('#fafafa')

        # Lanes
        for x in self.true_up_lanes + self.true_down_lanes:
            ax.plot([x, x], [0, self.height], color='#bdbdbd', linewidth=1.0, zorder=0)

        xs = [v.position[0] for v in self.vehicles]
        ys = [v.position[1] for v in self.vehicles]

        # BS/RSUs
        if getattr(self, "bs_positions", None):
            bx = [p[0] for p in self.bs_positions]
            by = [p[1] for p in self.bs_positions]
            ax.scatter(bx, by, c="#2e7d32", marker="s", s=80, edgecolors='black', linewidths=0.6, zorder=3)
            for i, (x, y) in enumerate(self.bs_positions):
                label = "BS" if (self.v2i_mode == "single") else f"B{i}"
                ax.text(x + 0.6, y, label, fontsize=7, va='center', color="#1b5e20", zorder=4)

        # Vehicles: compact directional triangles for large-map readability
        edge_map = {'u': '#d32f2f', 'd': '#1976d2'}
        fill_map = {'u': '#eef1f4', 'd': '#eef1f4'}
        for i, v in enumerate(self.vehicles):
            is_leader = (i == getattr(self, 'leader_idx_up', -1)) or (i == getattr(self, 'leader_idx_down', -1))
            center = np.asarray([float(v.position[0]), float(v.position[1])], dtype=np.float64)
            marker = '^' if str(getattr(v, 'direction', 'u')).lower() == 'u' else 'v'
            size = 52 if getattr(v, 'is_cav', True) else 40
            edge_lw = 1.2 if is_leader else 0.85
            if is_leader:
                ax.scatter(center[0], center[1],
                           s=255,
                           marker='o',
                           facecolors='none',
                           edgecolors='black',
                           linewidths=1.0,
                           alpha=0.18,
                           zorder=2.9)
            ax.scatter(
                center[0],
                center[1],
                s=size,
                marker=marker,
                c=fill_map.get(v.direction, '#eef1f4'),
                edgecolors=edge_map.get(v.direction, '#333333'),
                linewidths=edge_lw,
                alpha=0.98,
                zorder=3.05,
            )
            if is_leader:
                direction = str(getattr(v, 'direction', 'u')).lower()
                star_dy = 0.95 if direction == 'u' else -0.95
                ax.scatter(center[0], center[1] + star_dy, s=78, marker='*', c='#FFD166', edgecolors='black', linewidths=0.5, zorder=3.4)
                ax.text(center[0] + 0.75, center[1], f"L-{i}", fontsize=7, va='center', zorder=4)

        # Neighbor edges
        lines_main = []
        for i, v in enumerate(self.vehicles):
            for nb in getattr(v, 'neighbors', []):
                if i < nb:
                    lines_main.append([(xs[i], ys[i]), (xs[nb], ys[nb])])
        if lines_main:
            lc = LineCollection(lines_main, colors='#888888', linewidths=1.0, zorder=1)
            ax.add_collection(lc)

        # Destinations dashed
        if show_destinations:
            lines_dest = []
            for i, v in enumerate(self.vehicles):
                for d in getattr(v, 'destinations', []):
                    if 0 <= d < self.n_Veh and d != i:
                        lines_dest.append([(xs[i], ys[i]), (xs[d], ys[d])])
            if lines_dest:
                lcd = LineCollection(lines_dest, colors='tab:green', linewidths=0.6,
                                     linestyles='dashed', alpha=0.45, zorder=2)
                ax.add_collection(lcd)

        # V2I dashed (vehicle -> serving BS/RSU)
        if show_v2i and getattr(self, "v2i_serving_idx", None) is not None and len(self.bs_positions) > 0:
            lines_v2i = []
            for i in range(self.n_Veh):
                bi = int(self.v2i_serving_idx[i])
                if 0 <= bi < len(self.bs_positions):
                    bx, by = self.bs_positions[bi]
                    lines_v2i.append([(xs[i], ys[i]), (bx, by)])
            if lines_v2i:
                lcv = LineCollection(lines_v2i, colors='#9e9e9e', linewidths=0.7,
                                     linestyles='dashed', alpha=0.6, zorder=1)
                ax.add_collection(lcv)

        # Optional (RB, Power) annotations
        if annotate_power and agent is not None:
            acts = agent.action_all_with_power_training
            for i in range(min(self.n_Veh, acts.shape[0])):
                lines = []
                for j in range(3):
                    rb = int(acts[i, j, 0])
                    pw = int(acts[i, j, 1])
                    if rb >= 0:
                        lines.append(f"{j}:RB{rb}/P{pw}")
                if lines:
                    ax.text(xs[i] - 2.5, ys[i], "\n".join(lines),
                            fontsize=6, color='black', alpha=0.75, ha='right', va='center',
                            bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='gray', lw=0.4))

        # Fixed axes (map does not move)
        x_min = min(self.true_up_lanes + self.true_down_lanes) - 8
        x_max = max(self.true_up_lanes + self.true_down_lanes) + 8
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0, self.height)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        title = f'Dual 4-lane | {self.topology_type.upper()} per dir | N_up={self.n_up}, N_down={self.n_down}'
        if len(getattr(self, "bs_positions", [])) > 0:
            if self.v2i_mode == "single":
                title += ' | Single BS'
            else:
                title += f' | RSU:{self.bs_layout}, Δ={int(self.bs_spacing)}m, K={len(self.bs_positions)}'
        ax.set_title(title)
        ax.grid(alpha=0.25, linestyle='--')

        # Legend patches (optional)
        rsu_patch = mpatches.Patch(color='#2e7d32', label=('BS' if self.v2i_mode == "single" else 'RSU'), alpha=0.8)
        ax.legend(handles=[rsu_patch], loc='upper right', fontsize=8, frameon=True)

        import matplotlib.pyplot as plt2  # prevent shadowing
        plt.tight_layout()
        plt.savefig(save_path, dpi=dpi)
        plt.close(fig)
        print(f"[HighwayTopoEnv] Visualization saved -> {save_path}")
