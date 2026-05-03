import numpy as np
from rigid_body import RigidBody

class Joint:
    def __init__(self, body_a: RigidBody, body_b: RigidBody,
                min_angle: float, max_angle: float,
                max_torque: float, stiffness: float):

        self.body_a = body_a
        self.body_b = body_b
        self.min_angle = float(min_angle)
        self.max_angle = float(max_angle)
        self.max_torque = float(max_torque)
        self.stiffness = float(stiffness)
        self.angle = 0.0
        self.angular_velocity = 0.0
        self.torque_input = 0.0

    def apply_torque(self, value: np.float32):
        self.torque_input = float(np.clip(value, -1.0, 1.0))

    def step(self, dt: float):
        # 1 y 2: torque neto
        torque = self.torque_input * self.max_torque
        torque -= self.stiffness * self.angular_velocity

        # 3: inercia del cuerpo distal
        inertia = (self.body_b.mass * 
                (self.body_b.width ** 2 + self.body_b.height ** 2) / 12)
        inertia = max(inertia, 1.0)

        # 4: aceleración angular
        angular_acc = torque / inertia

        # 5: integrar
        self.angular_velocity += angular_acc * dt
        self.angle += self.angular_velocity * dt

        # 6: clippear y frenar si choca con límite
        if self.angle < self.min_angle:
            self.angle = self.min_angle
            self.angular_velocity = max(0.0, self.angular_velocity)
        elif self.angle > self.max_angle:
            self.angle = self.max_angle
            self.angular_velocity = min(0.0, self.angular_velocity)

        # 7: reposicionar body_b
        self._update_body_b_position()

    def _update_body_b_position(self):
        # Punto de anclaje: extremo inferior de body_a
        # En Pygame y aumenta hacia abajo, por eso sumamos height/2
        anchor_x = self.body_a.x
        anchor_y = self.body_a.y + self.body_a.height / 2

        # Centro de body_b: desde el anclaje, nos movemos en la dirección del ángulo
        # sin(angle) controla cuánto nos movemos horizontalmente
        # cos(angle) controla cuánto nos movemos verticalmente
        # Multiplicamos por height/2 porque el anclaje es el extremo, no el centro
        self.body_b.x = anchor_x + np.sin(self.angle) * (self.body_b.height / 2)
        self.body_b.y = anchor_y + np.cos(self.angle) * (self.body_b.height / 2)

        # Actualizamos el ángulo visual de body_b para que se dibuje inclinado
        self.body_b.angle = self.angle


