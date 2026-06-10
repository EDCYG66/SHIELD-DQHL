"""Motion-model utilities for straight-highway platoon simulation."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


class KinematicBicycleRoadModel:
    """Road-aligned kinematic bicycle model for straight highway driving.

    The longitudinal road axis is the global Y axis, while X denotes the
    lateral lane axis. For upward traffic, the vehicle progresses along +Y;
    for downward traffic, along -Y. The stored ``heading_error`` is the yaw
    deviation from the nominal lane direction, which keeps the model simple
    while still yielding steering-driven lane changes.
    """

    def __init__(
        self,
        *,
        min_speed_for_control: float = 1.5,
        heading_response: float = 1.8,
        max_heading_rate: float = 0.35,
        default_wheelbase: float = 2.75,
        truck_wheelbase: float = 5.20,
        default_vehicle_width: float = 1.85,
        truck_vehicle_width: float = 2.45,
        default_max_steer: float = 0.50,
        truck_max_steer: float = 0.34,
        default_max_steer_rate: float = 0.75,
        truck_max_steer_rate: float = 0.45,
        default_max_heading_error: float = 0.55,
    ):
        self.min_speed_for_control = float(min_speed_for_control)
        self.heading_response = float(heading_response)
        self.max_heading_rate = float(max_heading_rate)
        self.default_wheelbase = float(default_wheelbase)
        self.truck_wheelbase = float(truck_wheelbase)
        self.default_vehicle_width = float(default_vehicle_width)
        self.truck_vehicle_width = float(truck_vehicle_width)
        self.default_max_steer = float(default_max_steer)
        self.truck_max_steer = float(truck_max_steer)
        self.default_max_steer_rate = float(default_max_steer_rate)
        self.truck_max_steer_rate = float(truck_max_steer_rate)
        self.default_max_heading_error = float(default_max_heading_error)

    def initialize_vehicle(self, veh) -> None:
        """Populate the per-vehicle dynamic state used by the bicycle model."""

        length = float(getattr(veh, "vehicle_length", 4.8))
        is_truck = length > 7.5
        wheelbase = self.truck_wheelbase if is_truck else self.default_wheelbase
        vehicle_width = self.truck_vehicle_width if is_truck else self.default_vehicle_width
        max_steer = self.truck_max_steer if is_truck else self.default_max_steer
        max_steer_rate = self.truck_max_steer_rate if is_truck else self.default_max_steer_rate

        veh.wheelbase = float(getattr(veh, "wheelbase", wheelbase))
        veh.vehicle_width = float(getattr(veh, "vehicle_width", vehicle_width))
        veh.max_steer = float(getattr(veh, "max_steer", max_steer))
        veh.max_steer_rate = float(getattr(veh, "max_steer_rate", max_steer_rate))
        veh.max_heading_error = float(getattr(veh, "max_heading_error", self.default_max_heading_error))
        veh.heading_error = float(getattr(veh, "heading_error", 0.0))
        veh.steering_angle = float(getattr(veh, "steering_angle", 0.0))
        veh.yaw = float(getattr(veh, "yaw", self.global_yaw(getattr(veh, "direction", "u"), veh.heading_error)))

    def global_yaw(self, direction: str, heading_error: float) -> float:
        sign = self._direction_sign(direction)
        return float(sign * (0.5 * np.pi - float(heading_error)))

    def step_vehicle(
        self,
        veh,
        *,
        accel: float,
        desired_lateral_speed: float,
        dt: float,
        x_bounds: Tuple[float, float],
        y_bounds: Tuple[float, float],
    ) -> Dict[str, float]:
        """Advance one vehicle by one time step under the bicycle model."""

        self.initialize_vehicle(veh)
        dt = max(float(dt), 1e-4)
        direction = str(getattr(veh, "direction", "u"))
        sign = self._direction_sign(direction)
        wheelbase = max(float(getattr(veh, "wheelbase", self.default_wheelbase)), 1e-3)
        max_steer = abs(float(getattr(veh, "max_steer", self.default_max_steer)))
        max_steer_rate = abs(float(getattr(veh, "max_steer_rate", self.default_max_steer_rate)))
        max_heading_error = abs(float(getattr(veh, "max_heading_error", self.default_max_heading_error)))

        speed_prev = max(float(getattr(veh, "velocity", 0.0)), 0.0)
        speed_next = max(0.0, speed_prev + float(accel) * dt)
        speed_mid = 0.5 * (speed_prev + speed_next)

        heading_prev = float(getattr(veh, "heading_error", 0.0))
        steer_prev = float(getattr(veh, "steering_angle", 0.0))
        control_speed = max(speed_mid, self.min_speed_for_control)
        max_lateral_speed = control_speed * np.sin(max_heading_error)
        lateral_speed_cmd = float(np.clip(float(desired_lateral_speed), -max_lateral_speed, max_lateral_speed))
        desired_heading = float(np.arcsin(np.clip(lateral_speed_cmd / max(control_speed, 1e-6), -1.0, 1.0)))
        desired_heading_rate = float(
            np.clip(self.heading_response * (desired_heading - heading_prev), -self.max_heading_rate, self.max_heading_rate)
        )
        steer_target = float(np.arctan((wheelbase * desired_heading_rate) / max(control_speed, 1e-6)))
        steer_target = float(np.clip(steer_target, -max_steer, max_steer))
        steer_step = max_steer_rate * dt
        steer_next = float(np.clip(steer_target, steer_prev - steer_step, steer_prev + steer_step))

        heading_rate = float((speed_mid / wheelbase) * np.tan(steer_next))
        heading_next = float(np.clip(heading_prev + heading_rate * dt, -max_heading_error, max_heading_error))

        x_prev = float(veh.position[0])
        y_prev = float(veh.position[1])
        x_next = x_prev + speed_mid * np.sin(heading_next) * dt
        y_next = y_prev + sign * speed_mid * np.cos(heading_next) * dt

        x_min, x_max = float(x_bounds[0]), float(x_bounds[1])
        y_min, y_max = float(y_bounds[0]), float(y_bounds[1])
        x_clipped = float(np.clip(x_next, x_min, x_max))
        y_clipped = float(np.clip(y_next, y_min, y_max))

        if not np.isclose(x_clipped, x_next):
            heading_next *= 0.5
            steer_next *= 0.5
        if not np.isclose(y_clipped, y_next):
            speed_next = 0.0
            heading_next = 0.0
            steer_next = 0.0

        veh.position[0] = x_clipped
        veh.position[1] = y_clipped
        veh.velocity = float(speed_next)
        veh.heading_error = float(heading_next)
        veh.steering_angle = float(steer_next)
        veh.yaw = self.global_yaw(direction, heading_next)

        return {
            "speed": float(speed_next),
            "heading_error": float(heading_next),
            "steering_angle": float(steer_next),
            "desired_heading": float(desired_heading),
            "desired_lateral_speed": float(lateral_speed_cmd),
            "x": float(x_clipped),
            "y": float(y_clipped),
        }

    def _direction_sign(self, direction: str) -> float:
        return 1.0 if str(direction) == "u" else -1.0
