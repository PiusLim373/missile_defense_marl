#!/usr/bin/env python
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import numpy as np
import gymnasium as gym
import pygame

# Constants
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 1000
BASE_SIZE = 50
MISSILE_SIZE = 10
MAX_ACCELERATION = 0.02
MAX_SPEED = 2
DRONE_RADIUS = 300
DRONE_SIZE = 20
NUM_DRONES = 3

# Colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)

# Rewards
BASE_HIT_REWARD = -50
DRONE_HIT_REWARD = 100.0
STEP_PENALTY = -0.01
OUT_OF_BOUND_REWARD = -100

OBSERVATION_SPACE = gym.spaces.Box(low=np.array(
    [-1.0, -1.0, 0]), high=np.array([1.0, 1.0, 1500]), dtype=np.float32)
ACTION_SPACE = gym.spaces.Box(low=-10.0, high=10.0, shape=(2,))


class MissileDefenseEnv(MultiAgentEnv):
    def __init__(self,  config=None, render=False, realistic_render=False):
        self.render_flag = render
        self.realistic_render = realistic_render
        if self.render_flag:
            pygame.init()
            self.screen = pygame.display.set_mode(
                (SCREEN_WIDTH, SCREEN_HEIGHT))
            if self.realistic_render:
                self.base_png = pygame.image.load("./pygame_asset/base.png").convert_alpha()
                self.missile_png = pygame.image.load("./pygame_asset/missile.png").convert_alpha()
                self.drone_png = pygame.image.load("./pygame_asset/drone.png").convert_alpha()
            pygame.display.set_caption("Missile Defense System")
            self.clock = pygame.time.Clock()
        self.agents = [f"drone_{i}" for i in range(NUM_DRONES)]
        self._agent_ids = self.agents
        self.base_pos = np.array([SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2])
        self.observation_space = OBSERVATION_SPACE
        self.action_space = ACTION_SPACE

    def reset(self, seed=None, options=None):
        print("******reset*******")
        self.agents = [f"drone_{i}" for i in range(NUM_DRONES)]

        # Initialize missile
        if np.random.rand() > 0.5:
            x = np.random.choice([-MISSILE_SIZE, SCREEN_WIDTH + MISSILE_SIZE])
            y = np.random.uniform(0, SCREEN_HEIGHT)
        else:
            x = np.random.uniform(0, SCREEN_WIDTH)
            y = np.random.choice([-MISSILE_SIZE, SCREEN_HEIGHT + MISSILE_SIZE])
        self.missile_pos = np.array([x, y], dtype=np.float32)
        self.missile_velocity = np.zeros(2, dtype=np.float32)

        # Initialize drones in a circle around the base
        self.drones_pos = {
            agent: self.base_pos + DRONE_RADIUS *
            np.array([np.cos(theta), np.sin(theta)])
            for agent, theta in zip(self.agents, np.linspace(0, 2 * np.pi, NUM_DRONES, endpoint=False))
        }

        observations = {agent: self._get_obs(agent) for agent in self.agents}
        return observations, {}

    def step(self, actions):
        # Move missile towards base
        direction = self.base_pos - self.missile_pos
        distance = np.linalg.norm(direction)
        if distance > 0:
            direction /= distance
        self.missile_velocity += direction * MAX_ACCELERATION
        speed = np.linalg.norm(self.missile_velocity)
        if speed > MAX_SPEED:
            self.missile_velocity = self.missile_velocity / speed * MAX_SPEED
        self.missile_pos += self.missile_velocity

        # Move drones
        for agent, action in actions.items():
            if agent in self.drones_pos:
                self.drones_pos[agent] += action

        # Compute rewards, dones, and infos
        rewards = {agent: STEP_PENALTY for agent in self.agents}
        dones = {agent: False for agent in self.agents}
        dones["__all__"] = False

        # If drone went out of bounds, stop providing its stuffs
        for agent in list(self.agents):
            if np.any(self.drones_pos[agent] < 0) or np.any(self.drones_pos[agent] > SCREEN_WIDTH):
                rewards[agent] += OUT_OF_BOUND_REWARD
                print(
                    f"{agent} went out of bound at {self.drones_pos[agent]}, not providing any more observation")
                dones[agent] = True
                self.agents.remove(agent)
        if len(self.agents) == 0:
            print("All drones went out of bound, terminate")
            dones["__all__"] = True

        # Check for missile hitting base
        if np.linalg.norm(self.missile_pos - self.base_pos) < BASE_SIZE / 2:
            print("Base hit, game lost and terminate")
            dones["__all__"] = True
            rewards = {agent: BASE_HIT_REWARD for agent in self.agents}

        # Check for drone intercepting missile, terminate
        for agent in list(self.agents):
            if np.linalg.norm(self.drones_pos[agent] - self.missile_pos) < MISSILE_SIZE + DRONE_SIZE:
                rewards[agent] += DRONE_HIT_REWARD
                print("missile intercepted, threat eliminated")
                dones["__all__"] = True

        # update the observations, dont include the drones that went out of bound, else the training will crash
        observations = {agent: self._get_obs(agent) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        # we dont need this truncateds as the episode will terminate when base is hit
        truncateds = {agent: False for agent in self.agents}
        truncateds["__all__"] = False

        if self.render_flag:
            self.render()
        return observations, rewards, dones, truncateds, infos

    def _get_obs(self, agent):
        direction = self.missile_pos - self.drones_pos[agent]
        distance = np.linalg.norm(direction)

        if distance >= 0.0:
            unit_vector = direction / distance
            if np.any(unit_vector >= 1.0):
                print("unit vector is greater than 1.0, shouldnt happen!!!!!")
                print(f"unit_vector: {unit_vector}")
                print(f"direction: {direction}, distance: {distance}")
                print(
                    f"missile_pos: {self.missile_pos}, drone_pos: {self.drones_pos[agent]}")
        else:
            # To avoid division by zero, shouldnt happen as will get terminate earlier
            unit_vector = np.zeros(2)
        return np.concatenate((unit_vector, [distance])).astype(np.float32)

    def render(self):
        """
        (0,0)------------> +u (x) position 0
        |          (4,1)
        |
        |
        |
        v
        +v (y) position 1
        """
        self.screen.fill(WHITE)
        if self.realistic_render:
            base_rect = self.base_png.get_rect(center=self.base_pos)
            self.screen.blit(self.base_png, base_rect)
            missile_direction = self.missile_velocity
            angle = -np.degrees(np.arctan2(missile_direction[1], missile_direction[0]))
            rotated_missile_img = pygame.transform.rotate(self.missile_png, angle)
            missile_rect = rotated_missile_img.get_rect(center=self.missile_pos.astype(int))
            self.screen.blit(rotated_missile_img, missile_rect)
            for drone_pos in self.drones_pos.values():
                drone_rect = self.drone_png.get_rect(center=drone_pos.astype(int))
                self.screen.blit(self.drone_png, drone_rect)
        else:
            pygame.draw.rect(self.screen, BLUE, (*self.base_pos -
                            BASE_SIZE // 2, BASE_SIZE, BASE_SIZE))
            pygame.draw.circle(
                self.screen, RED, self.missile_pos.astype(int), MISSILE_SIZE)
            for drone_pos in self.drones_pos.values():
                pygame.draw.circle(self.screen, GREEN,
                                drone_pos.astype(int), DRONE_SIZE)
        pygame.display.flip()
        # the higher the number, the faster the simulation
        self.clock.tick(600)

    def close(self):
        if self.render_flag:
            pygame.quit()


if __name__ == "__main__":
    env = MissileDefenseEnv(render=True, realistic_render=True)
    obs = env.reset()
    done = False
    actions = {"drone_0": np.array([0.1, 0.1]), "drone_1": np.array(
        [-0.1, -0.1]), "drone_2": np.array([0.0, 0.0])}
    obs, rewards, terminateds, truncateds, infos = env.step(actions)
    while not done:
        actions = {"drone_0": np.array([0.1, 0.1]), "drone_1": np.array(
            [-0.1, -0.1]), "drone_2": np.array([0.0, 0.0])}
        obs, rewards, dones, truncateds, infos = env.step(actions)
        done = dones["__all__"]
    env.close()
