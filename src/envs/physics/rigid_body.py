import numpy as np

class RigidBody:

    def __init__(self, x: int, y: int, width: int, height: int,
                    mass: np.float32):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.mass = mass
        self.vx = 0.0
        self.vy = 0.0
        self.angle = 0.0
        self.angle_velocity = 0.0
        self.fx = 0.0
        self.fy = 0.0

    def apply_forces(self, fx: np.float32, fy: np.float32):
        self.fx += fx
        self.fy += fy

    def step(self, dt: np.float32, gravity: np.float32):
        self.fy += self.mass * gravity
        ax = self.fx / self.mass 
        ay = self.fy / self.mass
        self.vx += ax * dt
        self.vy += ay * dt
        self.x += self.vx * dt
        self.y += self.vy * dt 
        self.fx = 0.0
        self.fy = 0.0

    def get_corners(self):
        """Devuelve las 4 esquinas del cuerpo (sin rotación por ahora)."""
        hw = self.width / 2
        hh = self.height / 2
        return [
            (self.x - hw, self.y - hh),  # top-left
            (self.x + hw, self.y - hh),  # top-right
            (self.x + hw, self.y + hh),  # bottom-right
            (self.x - hw, self.y + hh),  # bottom-left
        ]

    def __repr__(self):
        return (f"RigidBody(pos=({self.x:.1f},{self.y:.1f}), "
                f"vel=({self.vx:.1f},{self.vy:.1f}))")
