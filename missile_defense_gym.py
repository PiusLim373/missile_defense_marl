#!/usr/bin/env python
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import numpy as np
import gymnasium as gym
import pygame
import random

np.set_printoptions(suppress=True)  # Disable scientific notation
np.set_printoptions(precision=3)
# Constants
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 1000
BASE_SIZE = 50
MISSILE_SIZE = 10
MAX_ACCELERATION = 0.02
MAX_SPEED = 0.5
DRONE_RADIUS = 300
DRONE_SIZE = 20
NUM_DRONES = 3
NUM_MISSILES = 3
CURRENT_MISSILE = 1
# Colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)

# Rewards
BASE_HIT_REWARD = -15
DRONE_HIT_REWARD = 40.0
STEP_PENALTY = 0.0
OUT_OF_BOUND_REWARD = -10
CLOSER_TO_MISSILE_REWARD = 0

OBSERVATION_SPACE = gym.spaces.Box(
    low=np.array([-10, -10, -1, -1, -1.1, -1.1, -1, -1, -1, -1.1, -1.1, -1, -1, -1, -1.1, -1.1, -1], dtype=np.float32),
    high=np.array(
        [
            SCREEN_WIDTH + 10,
            SCREEN_HEIGHT + 10,
            1000,
            1000,
            1.1,
            1.1,
            1500,
            1000,
            1000,
            1.1,
            1.1,
            1500,
            1000,
            1000,
            1.1,
            1.1,
            1500,
        ],
        dtype=np.float32,
    ),
    shape=(17,),
    dtype=np.float32,
)
ACTION_SPACE = gym.spaces.Box(low=-5.0, high=5.0, shape=(2,), dtype=np.int16)


class MissileDefenseEnv(MultiAgentEnv):
    def __init__(self, config=None, render=False, realistic_render=False):
        self.render_flag = render
        self.realistic_render = realistic_render
        if self.render_flag:
            pygame.init()
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            if self.realistic_render:
                self.base_png = pygame.image.load("./pygame_asset/base.png").convert_alpha()
                self.missile_png = pygame.image.load("./pygame_asset/missile.png").convert_alpha()
                self.drone_png = pygame.image.load("./pygame_asset/drone.png").convert_alpha()
                self.background_png = pygame.image.load("./pygame_asset/background.png").convert_alpha()
            pygame.display.set_caption("Missile Defense System")
            self.clock = pygame.time.Clock()
        self.agents = [f"drone_{i}" for i in range(NUM_DRONES)]
        self._agent_ids = self.agents
        self.base_pos = np.array([SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2])
        self.observation_space = OBSERVATION_SPACE
        self.action_space = ACTION_SPACE
        self.level = 0
        self.selected_level = 0
        self.experiences = {
            0: {"level_up_threshold": 1500, "solved_counter": 0},  # 1 missiles with speed cap
            1: {"level_up_threshold": 1500, "solved_counter": 0},  # 2 missiles with speed cap
            2: {"level_up_threshold": 1500, "solved_counter": 0},  # 3 missiles with speed cap
            3: {"level_up_threshold": 0, "solved_counter": 0},  # 3 missiles without speed cap
        }

    def reset(self, seed=None, options=None):
        global CURRENT_MISSILE, MAX_SPEED, MAX_ACCELERATION
        print("******reset*******")
        print(f"Level: {self.level}, experience: {self.experiences}")
        self.agents = [f"drone_{i}" for i in range(NUM_DRONES)]

        # check should level up:
        if (
            self.level != 3
            and self.experiences[self.level]["solved_counter"] >= self.experiences[self.level]["level_up_threshold"]
        ):
            self.level += 1
            print(f"Level up to {self.level}")

        self.selected_level = self.level
        if random.random() > 0.8:
            self.selected_level = random.randint(0, self.level)
        print(f"Selected level: {self.selected_level}")

        if self.selected_level == 0:
            CURRENT_MISSILE = 1
            MAX_SPEED = 0.5
        elif self.selected_level == 1:
            CURRENT_MISSILE = 2
            MAX_SPEED = 0.1
            MAX_ACCELERATION = 0.01
        elif self.selected_level == 2:
            CURRENT_MISSILE = 2
            MAX_SPEED = 0.5
            MAX_ACCELERATION = 0.02
        elif self.selected_level == 3:
            CURRENT_MISSILE = 2
            MAX_SPEED = 1.0
            MAX_ACCELERATION = 0.02

        # Initialize missile
        self.missiles_data = {
            f"missile_{missile_id}": {
                "missile_pos": np.array([-1, -1], dtype=np.float32),
                "missile_velocity": np.zeros(2, dtype=np.float32),
                "missile_acceleration": np.random.uniform(0, MAX_ACCELERATION),
                "neutralized": True,
            }
            for missile_id in range(NUM_MISSILES)
        }

        # Randomly select `CURRENT_MISSILE` missiles to be active
        active_missiles = random.sample(list(self.missiles_data.keys()), CURRENT_MISSILE)

        for missile_id in active_missiles:
            if np.random.rand() > 0.5:
                x = np.random.choice([0, SCREEN_WIDTH])
                y = np.random.uniform(0, SCREEN_HEIGHT)
            else:
                x = np.random.uniform(0, SCREEN_WIDTH)
                y = np.random.choice([0, SCREEN_HEIGHT])

            # Activate the missile
            self.missiles_data[missile_id] = {
                "missile_pos": np.array([x, y], dtype=np.float32),
                "missile_velocity": np.zeros(2, dtype=np.float32),
                "missile_acceleration": np.random.uniform(0, MAX_ACCELERATION),
                "neutralized": False,
            }

        # Initialize drones in a circle around the base
        self.drones_pos = {
            agent: self.base_pos + DRONE_RADIUS * np.array([np.cos(theta), np.sin(theta)])
            for agent, theta in zip(self.agents, np.linspace(0, 2 * np.pi, NUM_DRONES, endpoint=False))
        }
        self.drones_dist = {agent: 999.0 for agent in self.agents}

        observations = {agent: self._get_obs(agent) for agent in self.agents}
        return observations, {}

    def step(self, actions):
        # Updates missiles positions
        for missile_id, missile_data in self.missiles_data.items():
            if missile_data["neutralized"]:
                continue
            missile_pos = missile_data["missile_pos"]
            missile_velocity = missile_data["missile_velocity"]
            missile_acceleration = missile_data["missile_acceleration"]
            # Move missile towards the base (or target)
            direction = self.base_pos - missile_pos
            distance = np.linalg.norm(direction)
            if distance > 0:
                direction /= distance  # Normalize direction vector
                # prev_velocity = missile_velocity
                missile_velocity += direction * missile_acceleration
                speed = np.linalg.norm(missile_velocity)
                if speed > MAX_SPEED:
                    # missile_velocity = prev_velocity
                    missile_velocity = missile_velocity / speed * MAX_SPEED
            # Update missile position based on its velocity
            missile_pos += missile_velocity
            self.missiles_data[missile_id]["missile_pos"] = missile_pos
            self.missiles_data[missile_id]["missile_velocity"] = missile_velocity

        # Updates drones positions
        for agent, action in actions.items():
            if agent in self.drones_pos:
                self.drones_pos[agent] += action

        # Compute rewards, dones, and infos
        rewards = {agent: STEP_PENALTY for agent in self.agents}
        dones = {agent: False for agent in self.agents}
        dones["__all__"] = False

        # Check for missile hitting base
        for missile_id, missile_data in self.missiles_data.items():
            if np.linalg.norm(missile_data["missile_pos"] - self.base_pos) < BASE_SIZE / 2:
                print("Base hit, game lost and terminate")
                dones["__all__"] = True
                rewards = {agent: BASE_HIT_REWARD for agent in self.agents}

        # If drone went out of bounds, stop providing its stuffs
        for agent in list(self.agents):
            if np.any(self.drones_pos[agent] < 0) or np.any(self.drones_pos[agent] > SCREEN_WIDTH):
                rewards[agent] += OUT_OF_BOUND_REWARD
                print(f"{agent} went out of bound at {self.drones_pos[agent]}, not providing any more observation")
                dones[agent] = True
                self.agents.remove(agent)

        # Check for drone intercepting missile, terminate
        for agent in list(self.agents):
            min_dist_to_missile = self.drones_dist[agent]
            for missile_id, missile_data in self.missiles_data.items():
                if missile_data["neutralized"]:
                    continue
                dist = np.linalg.norm(self.drones_pos[agent] - missile_data["missile_pos"])
                if dist < min_dist_to_missile:
                    min_dist_to_missile = dist
                if dist < MISSILE_SIZE + DRONE_SIZE:
                    rewards[agent] += DRONE_HIT_REWARD
                    print(f"{agent} intercepted missile {missile_id}, threat eliminated")
                    dones[agent] = True
                    missile_data["neutralized"] = True
                    if agent in self.agents:
                        self.agents.remove(agent)
            if min_dist_to_missile < self.drones_dist[agent]:
                self.drones_dist[agent] = min_dist_to_missile
                rewards[agent] += CLOSER_TO_MISSILE_REWARD

        remaining_missiles = sum([1 for missile_data in self.missiles_data.values() if not missile_data["neutralized"]])
        if remaining_missiles == 0:
            print("All missiles neutralized, terminate")
            self.experiences[self.selected_level]["solved_counter"] += 1
            dones["__all__"] = True

        if len(self.agents) == 0:
            print("Not more agent left, terminate")
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
        obs = [self.drones_pos[agent][0], self.drones_pos[agent][1]]
        for missile_id, missile_data in self.missiles_data.items():
            if missile_data["neutralized"]:
                obs.extend([-1, -1, 0, 0, -1])
            else:
                missile_pos = missile_data["missile_pos"]
                direction = missile_pos - self.drones_pos[agent]
                distance = np.linalg.norm(direction)
                if distance >= 0.0:
                    unit_vector = direction / distance
                else:
                    unit_vector = np.zeros(2)
                obs.extend([missile_pos[0], missile_pos[1]] + unit_vector.tolist() + [float(distance)])
        return np.array(obs)

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
            background_rect = self.background_png.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
            self.screen.blit(self.background_png, background_rect)
            base_rect = self.base_png.get_rect(center=self.base_pos)
            self.screen.blit(self.base_png, base_rect)
            for missile_data in self.missiles_data.values():
                missile_direction = missile_data["missile_velocity"]
                angle = -np.degrees(np.arctan2(missile_direction[1], missile_direction[0]))
                rotated_missile_img = pygame.transform.rotate(self.missile_png, angle)
                missile_rect = rotated_missile_img.get_rect(center=missile_data["missile_pos"].astype(int))
                self.screen.blit(rotated_missile_img, missile_rect)
            for drone_pos in self.drones_pos.values():
                drone_rect = self.drone_png.get_rect(center=drone_pos.astype(int))
                self.screen.blit(self.drone_png, drone_rect)
        else:
            pygame.draw.rect(self.screen, BLUE, (*self.base_pos - BASE_SIZE // 2, BASE_SIZE, BASE_SIZE))
            for missile_data in self.missiles_data.values():
                pygame.draw.circle(self.screen, RED, missile_data["missile_pos"], MISSILE_SIZE)
            for drone_pos in self.drones_pos.values():
                pygame.draw.circle(self.screen, GREEN, drone_pos.astype(int), DRONE_SIZE)
        pygame.display.flip()
        # the higher the number, the faster the simulation
        self.clock.tick(1000)

    def close(self):
        if self.render_flag:
            pygame.quit()


if __name__ == "__main__":
    env = MissileDefenseEnv(render=True, realistic_render=True)
    obs = env.reset()
    done = False
    # actions = {"drone_0": np.array([0.1, 0.1]), "drone_1": np.array([-0.1, -0.1]), "drone_2": np.array([0.0, 0.0])}
    # obs, rewards, terminateds, truncateds, infos = env.step(actions)
    while not done:
        # actions = {"drone_0": np.array([0.1, 0.1]), "drone_1": np.array([-0.1, -0.1]), "drone_2": np.array([0.0, 0.0])}
        actions = {"drone_0": np.array([0.0, 0.0]), "drone_1": np.array([-0.0, -0.0]), "drone_2": np.array([0.0, 0.0])}
        obs, rewards, dones, truncateds, infos = env.step(actions)
        done = dones["__all__"]
        print(obs)
        # input("enter")
    env.close()
