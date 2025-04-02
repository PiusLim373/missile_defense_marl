#!/usr/bin/env python
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import numpy as np
import gymnasium as gym
import pygame
import random
import copy
from icecream import ic

np.set_printoptions(suppress=True)  # Disable scientific notation
np.set_printoptions(precision=3, linewidth=200)
# Constants
TIME_STEP = 1
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 1000
BASE_SIZE = 50
MISSILE_SIZE = 10
MAX_MISSILE_ACCELERATION = 0.02
MAX_MISSILE_SPEED = 2.0
NUM_MISSILES = 3
CURRENT_MISSILE = 1
MAX_DRONE_VX = SCREEN_WIDTH / TIME_STEP
MAX_DRONE_VY = SCREEN_HEIGHT / TIME_STEP
MAX_DRONE_AX = MAX_DRONE_VX / TIME_STEP
MAX_DRONE_AY = MAX_DRONE_VY / TIME_STEP
DRONE_RADIUS = 300
DRONE_SIZE = 20
NUM_DRONES = 3
VELOCITY_THRESHOLD = 0
ACCELERATION_THRESHOLD = 0
VELOCITY_CHANGE_PENALTY_WEIGHT = -0.05
ACCELERATION_CHANGE_PENALTY_WEIGHT = -0.1
SMOOTHNESS_DISTANCE_THRESHOLD = 0
# MISSILE_COOLDOWN = 100

# Observation size
OBSERVATION_PER_DRONE = 6
BOUNDARY_OBSERVATION = 4
OBSERVATION_PER_MISSILE = 6

# Colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)

# Rewards
BASE_HIT_REWARD = -100
DRONE_HIT_REWARD = 200.0
STEP_PENALTY = 0.0
OUT_OF_BOUND_REWARD = -100
BOUNDARY_MARGIN = 50
BOUNDARY_MARGIN_MULTIPLIER = 5.0
CLOSER_TO_MISSILE_REWARD = 0

OBSERVATION_SPACE = gym.spaces.Box(
    low=np.array([*([-10, -10, -MAX_DRONE_VX, -MAX_DRONE_VY, -MAX_DRONE_AX, -MAX_DRONE_AY] * 1), # agent states
                  -10, -10, -10, -10, # boundary distances
                  *([-1, -1, -1.1, -1.1, -1, -10]*1)], # missile states
                  dtype=np.float32),
    high=np.array([*([SCREEN_WIDTH+10, SCREEN_HEIGHT+10, MAX_DRONE_VX, MAX_DRONE_VY, MAX_DRONE_AX, MAX_DRONE_AY] * 1), # agent states
                  SCREEN_WIDTH, SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_HEIGHT, # boundary distances
                   *([1000, 1000, 1.1, 1.1, 1500, np.linalg.norm([SCREEN_WIDTH+10, SCREEN_HEIGHT+10])] * 1)], # missile states
                   dtype=np.float32,),
    shape=(OBSERVATION_PER_DRONE*1 + BOUNDARY_OBSERVATION + OBSERVATION_PER_MISSILE*1,),
    dtype=np.float32,
)
ACTION_SPACE = gym.spaces.Box(low=-5.0, high=5.0, shape=(2,), dtype=np.int16)


class MissileDefenseEnv(MultiAgentEnv):
    def __init__(self, config=None, test_level=0, render=False, realistic_render=False):
        self.render_flag = render
        self.realistic_render = realistic_render
        self.dt = 1 # assuming every step forward is time unit of 1
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
        self.level = test_level
        self.selected_level = test_level
        self.experiences = {
            0: {"level_up_threshold": 200, "solved_counter": 0},  # 1 missiles with speed cap
            1: {"level_up_threshold": 200, "solved_counter": 0},  # 2 missiles with speed cap
            2: {"level_up_threshold": 200, "solved_counter": 0},  # 3 missiles with speed cap
            3: {"level_up_threshold": 0, "solved_counter": 0},  # 3 missiles without speed cap
        }
        self.MISSILE_COOLDOWN = 100 
        self.MAX_MISSILE_SPEED = 1.0  # Change this to a global variable if needed
        self.step_counter = 0
        self.level = 0
        self.missiles_intercepted  = 0
        self.survival_time = 0  
        self.LEVEL_UP_THRESHOLD = {
            0 : {"interceptions_needed": 1, "survival_steps": 4000},
            1 : {"interceptions_needed": 2, "survival_steps": 8000},
            2 : {"interceptions_needed": 3, "survival_steps": 12000},
            3 : None
        }

    def reset(self, seed=None, options=None):
        self.survival_time = 0
        self.missiles_intercepted = 0
        global CURRENT_MISSILE, MAX_MISSILE_SPEED, MAX_MISSILE_ACCELERATION
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
            MAX_MISSILE_SPEED = 0.5
        elif self.selected_level == 1:
            CURRENT_MISSILE = 2
            MAX_MISSILE_SPEED = 0.1
            MAX_MISSILE_ACCELERATION = 0.01
        elif self.selected_level == 2:
            CURRENT_MISSILE = 2
            MAX_MISSILE_SPEED = 0.5
            MAX_MISSILE_ACCELERATION = 0.02
        elif self.selected_level == 3:
            CURRENT_MISSILE = 2
            MAX_MISSILE_SPEED = 1.0
            MAX_MISSILE_ACCELERATION = 0.02

        # Initialize missile
        self.missile_counter = 0
        self.missiles_data = {
            f"missile_{missile_id}": {
                "id": self.missile_counter,
                "missile_pos": np.array([-1, -1], dtype=np.float32),
                "missile_velocity": np.zeros(2, dtype=np.float32),
                "missile_acceleration": np.random.uniform(0, MAX_MISSILE_ACCELERATION),
                "neutralized": True,
                "cooldown": 0 # Cooldown for missile
            }
            for missile_id in range(NUM_MISSILES)
        }
        self.missile_slots = [None]*NUM_MISSILES # store the missile id
        # Randomly select `CURRENT_MISSILE` missiles to be active
        active_missiles = random.sample(list(self.missiles_data.keys()), 3)
        for missile_id in active_missiles:
            self._spawn_missile(missile_id)
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
                "missile_acceleration": np.random.uniform(0, MAX_MISSILE_ACCELERATION),
                "neutralized": False,
            }

        # Initialize drones in a circle around the base
        self.drones_pos = {
            agent: self.base_pos + DRONE_RADIUS * np.array([np.cos(theta), np.sin(theta)])
            for agent, theta in zip(self.agents, np.linspace(0, 2 * np.pi, NUM_DRONES, endpoint=False))
        }
        self.prev_drones_pos = copy.copy(self.drones_pos)
        self.drones_dist = {agent: 999.0 for agent in self.agents}

        self.drones_vel = {agent: np.zeros(2) for agent in self.agents}
        self.prev_drones_vel = copy.copy(self.drones_vel)
        self.drones_acc = {agent: np.zeros(2) for agent in self.agents}
        self.prev_drones_acc = copy.copy(self.drones_acc)

        self.drone_target_ids = {agent: None for agent in self.agents}

        observations = {agent: self._get_obs(agent) for agent in self.agents}
        return observations, {}
    

    def _spawn_missile(self, missile_id): # FOR RESPAWN

        if np.random.rand() > 0.5:
            x = np.random.choice([0, SCREEN_WIDTH])
            y = np.random.uniform(0, SCREEN_HEIGHT)
        else:
            x = np.random.uniform(0, SCREEN_WIDTH)
            y = np.random.choice([0, SCREEN_HEIGHT])
        


        self.missiles_data[missile_id] = {
            "missile_pos": np.array([x, y], dtype=np.float32),
            "missile_velocity": np.zeros(2, dtype=np.float32),
            "missile_acceleration": np.random.uniform(0, MAX_MISSILE_ACCELERATION),
            "neutralized": False,
            "cooldown": 0 # Cooldown for missile
        }

        assert not self.missiles_data[missile_id]["neutralized"], f"Missile {missile_id} failed to respawn"
        print(f"Missile {missile_id} respawned at {self.missiles_data[missile_id]['missile_pos']}")
    
         
        

    
        


    def _calculate_rewards(self):
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
        # Define a threshold distance where penalty starts (e.g., 50 pixels from the boundary)
        # Find the closest boundary
        min_distance = min(self._get_agent_dist_from_boundary(agent))
        # Apply penalty if within SAFE_MARGIN
        if min_distance < BOUNDARY_MARGIN:
            boundary_penalty = -BOUNDARY_MARGIN_MULTIPLIER * (1 - min_distance / BOUNDARY_MARGIN)
            rewards[agent] += boundary_penalty

        # Check for drone intercepting missile, terminate
        for agent in list(self.agents):
            agent_pos = self.drones_pos[agent]
            agent_vel = self.drones_vel[agent]
            agent_accel = self.drones_acc[agent]


            for missile_id, missile_data in self.missiles_data.items():
                if missile_data["neutralized"]:
                    continue
                
                missile_pos = missile_data["missile_pos"]
                direction = missile_pos - agent_pos
                distance = np.linalg.norm(direction)
                # Normalize direction vector
                if distance > 0:
                    direction_unit = direction / distance
                else:
                    direction_unit = np.zeros(2)

                velocity_reward = np.dot(agent_vel, direction_unit)
                acceleration_reward = np.dot(agent_accel, direction_unit)
                rewards[agent] += 3.0 * velocity_reward + 3.0 * acceleration_reward
                # Distance-based reward (closer is better)
                # rewards[agent] += 1.0 / (distance + 1e-3)

                # Reward for interception
                if distance < MISSILE_SIZE + DRONE_SIZE:
                        self.missiles_intercepted += 1
                        rewards[agent] += DRONE_HIT_REWARD
                        
                        missile_data["neutralized"] = True
                        missile_data["cooldown"] = self.MISSILE_COOLDOWN  # Start countdown
                        print(f"{agent} intercepted missile, respawning in {self.MISSILE_COOLDOWN} steps")
                        #dones[agent] = True
                        #if agent in self.agents:
                           # self.agents.remove(agent)
                  

                # Penalty for missiles nearing the base
                missile_base_dist = np.linalg.norm(missile_pos - self.base_pos)
                urgency_penalty = -50.0 * (1 / (1 + missile_base_dist))
                rewards[agent] += urgency_penalty

        remaining_missiles = sum([1 for missile_data in self.missiles_data.values() if not missile_data["neutralized"]])
        #if remaining_missiles == 0:
         #   print("All missiles neutralized, terminate")
           # self.experiences[self.selected_level]["solved_counter"] += 1
          #  dones["__all__"] = True
        for missile_data in self.missiles_data.values():
            if np.linalg.norm(missile_data["missile_pos"] - self.base_pos) < BASE_SIZE / 2:
                print("Base hit, game lost and terminate")
                dones["__all__"] = True
                

        if len(self.agents) == 0:
            print("No agent left, terminate")
            dones["__all__"] = True
        return rewards, dones

    def step(self, actions):
        self.survival_time += 1
        self.step_counter += 1
        respawned_missiles = []
        current_threshold = self.LEVEL_UP_THRESHOLD[self.level]
        if self.level in self.LEVEL_UP_THRESHOLD:
            current_threshold = self.LEVEL_UP_THRESHOLD[self.level]
        

        if current_threshold and  (
            self.missiles_intercepted >= current_threshold["interceptions_needed"]
            and self.survival_time >= current_threshold["survival_steps"]
        ) :
            #if self.missiles_intercepted >= current_threshold["interceptions_needed"] and self.survival_time >= current_threshold["survival_steps"]:
                print("Level up!")
                self.level += 1
                print(f"Level up to {self.level}")
                self._increase_difficulty()

        for missile_id, missile_data in self.missiles_data.items():
            if missile_data["neutralized"]:
                # For neutralized missiles, update cooldown and possibly respawn.
                if missile_data.get("cooldown", 0) > 0:
                    missile_data["cooldown"] -= 1
                elif missile_data.get("cooldown", 0) == 0:
                    self._spawn_missile(missile_id)
                    respawned_missiles.append(missile_id)
        if respawned_missiles:
            print(f"Respawned missiles: {respawned_missiles} at step{self.step_counter}")
        
        # Update missiles positions
        for missile_id, missile_data in self.missiles_data.items():
            if missile_data["neutralized"]:
                # For neutralized missiles, update cooldown and possibly respawn.
                if missile_data.get("cooldown", 0) > 0:
                    missile_data["cooldown"] -= 1
                else:
                    self._spawn_missile(missile_id)
            else:
                # For active missiles update position
                missile_pos = missile_data["missile_pos"]
                missile_velocity = missile_data["missile_velocity"]
                missile_acceleration = missile_data["missile_acceleration"]
                # Move missile towards the base (or target)
                direction = self.base_pos - missile_pos
                distance = np.linalg.norm(direction)
                if distance > 0:
                    direction /= distance  # Normalize direction vector
                    missile_velocity += direction * missile_acceleration
                    speed = np.linalg.norm(missile_velocity)
                    if speed > self.MAX_MISSILE_SPEED:  # --> Change these instances to global to default and remove keyboard input
                        missile_velocity = missile_velocity / speed * self.MAX_MISSILE_SPEED
                # Update missile position based on its velocity
                missile_pos += missile_velocity
                self.missiles_data[missile_id]["missile_pos"] = missile_pos
                self.missiles_data[missile_id]["missile_velocity"] = missile_velocity

        # Update stored states
        for agent in self.agents:
            self.prev_drones_pos[agent] = np.copy(self.drones_pos[agent])
            self.prev_drones_vel[agent] = np.copy(self.drones_vel[agent])
            self.prev_drones_acc[agent] = np.copy(self.drones_acc[agent])
        for agent, action in actions.items():
            if agent in self.drones_pos:
                self.drones_pos[agent] += action

        # Now update velocity and acceleration
        for agent in self.agents:
            self.drones_vel[agent] = (self.drones_pos[agent] - self.prev_drones_pos[agent]) / self.dt
            self.drones_acc[agent] = (self.drones_vel[agent] - self.prev_drones_vel[agent]) / self.dt

        rewards, dones = self._calculate_rewards()
        #Smoothness reward
        for agent in self.agents:
            delta_vel = np.linalg.norm(self.drones_vel[agent] - self.prev_drones_vel[agent])
            delta_acc = np.linalg.norm(self.drones_acc[agent] - self.prev_drones_acc[agent])

            # Check distance to nearest active missile
            min_dist_to_missile = float("inf")
            for missile_data in self.missiles_data.values():
                if missile_data["neutralized"]:
                    continue
                dist = np.linalg.norm(self.drones_pos[agent] - missile_data["missile_pos"])
                min_dist_to_missile = min(min_dist_to_missile, dist)

            scale = 0.5 if min_dist_to_missile < SMOOTHNESS_DISTANCE_THRESHOLD else 1.0

            if delta_vel > VELOCITY_THRESHOLD:
                rewards[agent] += scale * VELOCITY_CHANGE_PENALTY_WEIGHT * (delta_vel - VELOCITY_THRESHOLD)
            if delta_acc > ACCELERATION_THRESHOLD:
                rewards[agent] += scale * ACCELERATION_CHANGE_PENALTY_WEIGHT * (delta_acc - ACCELERATION_THRESHOLD)

        #Debug log
        if self.agents:  # Only if there are agents remaining
            avg_dv = np.mean([np.linalg.norm(self.drones_vel[a] - self.prev_drones_vel[a]) for a in self.agents])
            avg_da = np.mean([np.linalg.norm(self.drones_acc[a] - self.prev_drones_acc[a]) for a in self.agents])
            print(f"[Smoothness] Avg Δv: {avg_dv:.2f}, Avg Δa: {avg_da:.2f}")
        # update the observations, dont include the drones that went out of bound, else the training will crash
        observations = {agent: self._get_obs(agent) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        # we dont need this truncateds as the episode will terminate when base is hit
        truncateds = {agent: False for agent in self.agents}
        truncateds["__all__"] = False

        if self.render_flag:
            self.render()
        return observations, rewards, dones, truncateds, infos

    def _get_agent_dist_from_boundary(self, agent):
        agent_x, agent_y = self.drones_pos[agent]
        dist_left = agent_x
        dist_right = SCREEN_WIDTH - agent_x
        dist_top = SCREEN_HEIGHT - agent_y
        dist_bottom = agent_y
        return [dist_left, dist_right, dist_top, dist_bottom]
    
    def _increase_difficulty(self):   # increase the difficulty of the game as per new parameters
        global MAX_MISSILE_SPEED, MAX_MISSILE_ACCELERATION 
        if self.level >= 3:
            return

           
        
        MAX_MISSILE_ACCELERATION  *= 1.1
        MAX_MISSILE_SPEED *= 1.2
        self.MISSILE_COOLDOWN = max(0, self.MISSILE_COOLDOWN - 10)  # Decrease cooldown





    def _get_obs(self, agent):
        obs = [self.drones_pos[agent][0], self.drones_pos[agent][1]]
        obs.extend(self.drones_vel[agent].tolist())   # vx, vy
        obs.extend(self.drones_acc[agent].tolist())  # ax, ay
        obs.extend(self._get_agent_dist_from_boundary(agent))
        # Compute and append other drones' velocities and accelerations
        # for other_agent in self._agent_ids:
        #     if other_agent == agent:
        #         continue  # Skip self
        #     obs.extend(self.drones_pos[other_agent].tolist())
        #     other_velocity = (self.drones_pos[other_agent] - self.prev_drones_pos[other_agent]) / self.dt
        #     other_acceleration = (other_velocity - self.prev_drones_vel[other_agent]) / self.dt
        #     obs.extend(other_velocity.tolist())   # Other agent velocity: [vx, vy]
        #     obs.extend(other_acceleration.tolist())  # Other agent acceleration: [ax, ay]

        # Compute distances and store them with missile data
        missile_list = []
        distance_to_agent = None
        for missile_id, missile_data in self.missiles_data.items():
            if missile_data["neutralized"]:
                missile_list.append((float('inf'), None, [-1, -1, 0, 0, -1, 999]))  # Neutralized missiles are treated as far away
            else:
                missile_pos = missile_data["missile_pos"]
                direction = missile_pos - self.drones_pos[agent]
                distance_to_agent = np.linalg.norm(direction)
                unit_vector = direction / distance_to_agent if distance_to_agent > 0 else np.zeros(2)
                missile_to_base_dist = np.linalg.norm(missile_pos - self.base_pos)  
                missile_list.append((
                    distance_to_agent,
                    missile_id,
                    [missile_pos[0], missile_pos[1]] + unit_vector.tolist() + [distance_to_agent, missile_to_base_dist]
                ))

        # Sort by distance (smallest first)
        missile_list.sort(key=lambda x: x[0])
        obs.extend(missile_list[0][2])

        # Assign nearest missile to current agent
        if missile_list[0][1] != self.drone_target_ids[agent]:
            print(f"Assigned {missile_list[0][1]} to {agent}")
            self.drone_target_ids[agent] = missile_list[0][1]

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
    env.close()
