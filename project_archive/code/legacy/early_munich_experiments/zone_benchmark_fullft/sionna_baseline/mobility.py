import numpy as np


class RandomWaypoint:
    """
    Random Waypoint Mobility Model for independent node movement.
    """

    def __init__(
        self,
        bounds_x=(-400, 400),
        bounds_y=(-400, 400),
        velocity=(1.0, 3.0),
        pause_time=2.0,
    ):
        self.bounds_x = bounds_x
        self.bounds_y = bounds_y
        self.velocity_range = velocity
        self.pause_time_max = pause_time

        self.destinations = {}  # node_id -> [x, y]
        self.velocities = {}  # node_id -> speed
        self.pause_timers = {}  # node_id -> time left

    def _get_random_destination(self):
        return np.array([np.random.uniform(*self.bounds_x), np.random.uniform(*self.bounds_y)])

    def step(self, positions, dt=1.0):
        """
        positions: dict of {node_id: np.array([x, y, z])}
        updates positions in place based on the timestep dt.
        """
        for node_id, pos in positions.items():
            if node_id not in self.destinations:
                self.destinations[node_id] = self._get_random_destination()
                self.velocities[node_id] = np.random.uniform(*self.velocity_range)
                self.pause_timers[node_id] = 0

            if self.pause_timers[node_id] > 0:
                self.pause_timers[node_id] -= dt
                continue

            dest = self.destinations[node_id]
            pos_2d = pos[:2]
            direction = dest - pos_2d
            distance = np.linalg.norm(direction)

            step_dist = self.velocities[node_id] * dt
            if step_dist >= distance:
                positions[node_id][:2] = dest
                self.destinations[node_id] = self._get_random_destination()
                self.velocities[node_id] = np.random.uniform(*self.velocity_range)
                self.pause_timers[node_id] = np.random.uniform(0, self.pause_time_max)
            else:
                direction /= distance
                positions[node_id][:2] = pos_2d + direction * step_dist

