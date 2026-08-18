"""
node.py
=======
Defines the Node class representing a radio-enabled device on the map.
"""

class Node:
    """
    A radio node located on a 2D plane with transmit power properties.

    Attributes
    ----------
    x : float
        The X-coordinate of the node (m).
    y : float
        The Y-coordinate of the node (m).
    z : float
        The Z-coordinate (height) of the node (m). Defaults to 1.5m.
    transmit_power : float
        The transmit power of the node in dBm. Defaults to 15.0 dBm.
    """

    def __init__(self, x: float, y: float, transmit_power_dbm: float = 15.0, z: float = 1.5):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.transmit_power = float(transmit_power_dbm)

    def move(self, dx, dy, walls, map_size):
        """
        Move the node by (dx, dy) if the new position is within the map
        boundaries and does not collide with any walls.
        """
        new_x = self.x + dx
        new_y = self.y + dy
        
        # Stay within map bounds (with small buffer)
        if not (0.2 <= new_x <= map_size - 0.2 and 0.2 <= new_y <= map_size - 0.2):
             return False 
             
        # Check wall collisions
        for name, cx, cy, th, l, h in walls:
             if (cx - th/2 - 0.1 <= new_x <= cx + th/2 + 0.1) and (cy - l/2 - 0.1 <= new_y <= cy + l/2 + 0.1):
                 return False 
        
        self.x, self.y = float(new_x), float(new_y)
        return True

    @property
    def position(self):
        """Returns the (x, y, z) position as a list for Sionna compatibility."""
        return [float(self.x), float(self.y), float(self.z)]

    def __repr__(self):
        return (f"Node(x={self.x:.2f}, y={self.y:.2f}, "
                f"z={self.z:.2f}, pwr={self.transmit_power} dBm)")
