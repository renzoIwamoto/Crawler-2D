from src.envs.physics.joint import Joint
from src.envs.physics.rigid_body import RigidBody

import pygame
import numpy as np


# Pantalla
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720

# Física
GRAVITY = 980.0
DT = 1/60

# Suelo
GROUND_Y = 600

# Dimensiones de los cuerpos
TORSO_W, TORSO_H = 100, 30
MUSLO_W, MUSLO_H = 15, 45
TIBIA_W, TIBIA_H = 12, 40

class CrawlerEnv:
    def __init__(self):
        self.trunk = RigidBody(200, 100, TORSO_W, TORSO_H, 8.0)

        # Muslo izquierdo: su centro está MUSLO_H/2 debajo del extremo inferior del torso
        # El extremo inferior del torso está en: torso.y + TORSO_H/2
        muslo_L_x = self.trunk.x - TORSO_W/4   # un poco a la izquierda del centro
        muslo_L_y = self.trunk.y + TORSO_H/2 + MUSLO_H/2
        self.thigh_L = RigidBody(muslo_L_x, muslo_L_y, MUSLO_W, MUSLO_H, 2.0)

        # Muslo derecho
        muslo_R_x = self.trunk.x + TORSO_W/4   # un poco a la derecha del centro
        muslo_R_y = self.trunk.y + TORSO_H/2 + MUSLO_H/2
        self.thigh_R = RigidBody(muslo_R_x, muslo_R_y, MUSLO_W, MUSLO_H, 2.0)

        shin_L_x = self.trunk.x - TORSO_W/4   # un poco a la izquierda del centro
        shin_L_y = self.thigh_L.y + TIBIA_H/2 + MUSLO_H/2
        self.shin_L = RigidBody(shin_L_x, shin_L_y, TIBIA_W, TIBIA_H, 2.0)        

        shin_R_x = self.trunk.x + TORSO_W/4   # un poco a la izquierda del centro
        shin_R_y = self.thigh_L.y + TIBIA_H/2 + MUSLO_H/2
        self.shin_R = RigidBody(shin_R_x, shin_R_y, TIBIA_W, TIBIA_H, 2.0)       

        self.hip_L = Joint(self.trunk, self.thigh_L, -np.pi/3, np.pi/3, 500.0, 0.1)
        self.knee_L = Joint(self.thigh_L, self.shin_L, 0, np.pi/2, 500.0, 0.1)
        self.hip_R = Joint(self.trunk, self.thigh_R, -np.pi/3, np.pi/3, 500.0, 0.1)
        self.knee_R = Joint(self.thigh_R, self.shin_R, 0, np.pi/2, 500.0, 0.1)

        self.ground_y = GROUND_Y
        self.dt = DT
        self.step_count = 0
        self.max_steps = 1000

        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode((1280, 720))
        self.clock = pygame.time.Clock()