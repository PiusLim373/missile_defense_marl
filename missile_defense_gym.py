#!/usr/bin/env python
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import numpy as np
import gymnasium as gym
import pygame
import random
import copy
import heapq

np.set_printoptions(suppress=True)  # Disable scientific notation
np.set_printoptions(precision=3, linewidth=200)

# Number of agents
NUM_DRONES = 5
NUM_MISSILES = 5

# Missile parameters
MAX_MISSILE_ACCELERATION = 0.2
MAX_MISSILE_SPEED = 1.0
MISSILE_COOLDOWN = 10
LEVEL_ACC_MULTIPLIER = 1.014
LEVEL_VEL_MULTIPLIER = 1.014
LEVEL_MISSILE_MULTIPLIER = 1.014
LEVEL_SURVIVAL_MULTIPLIER = 1500
# Constants
TIME_STEP = 1
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 1000
BASE_SIZE = 50
MISSILE_SIZE = 10
CURRENT_MISSILE = 1
MAX_DRONE_VX = SCREEN_WIDTH / TIME_STEP
MAX_DRONE_VY = SCREEN_HEIGHT / TIME_STEP
MAX_DRONE_AX = MAX_DRONE_VX / TIME_STEP
MAX_DRONE_AY = MAX_DRONE_VY / TIME_STEP
MAX_DISTANCE_FROM_BASE = np.linalg.norm([SCREEN_HEIGHT/2, SCREEN_WIDTH/2])
DRONE_RADIUS = 300
DRONE_SIZE = 20
VELOCITY_THRESHOLD = 0
ACCELERATION_THRESHOLD = 0
VELOCITY_CHANGE_PENALTY_WEIGHT = -0.2
ACCELERATION_CHANGE_PENALTY_WEIGHT = -0.4
SMOOTHNESS_DISTANCE_THRESHOLD = 20
MAX_VELOCITY = 5.0
MAX_ACCELERATION = 1.0
RETARGET_DISTANCE = 400

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
BASE_HIT_REWARD = -1000
DRONE_HIT_REWARD = 200.0
STEP_PENALTY = 0.0
ALIVE_REWARD = 0.5
OUT_OF_BOUND_REWARD = -1000
NEAR_BOUND_PENALTY = -5
BOUND_RADIUS = (min(SCREEN_HEIGHT, SCREEN_WIDTH) - DRONE_SIZE * 2) / 2
BOUNDARY_MARGIN_MULTIPLIER = 5.0
CLOSER_TO_MISSILE_REWARD = 0
APPROACH_VEL_REWARD = 5.0
APPROACH_ACC_REWARD = 5.0

OBSERVATION_SPACE = gym.spaces.Box(
    low=np.array(
        [
            -10, -10, -MAX_DRONE_VX, -MAX_DRONE_VY, -MAX_DRONE_AX, -MAX_DRONE_AY,  # agent states
            -1, -1, -1.1, -1.1, -1, 0, # assigned missile
            -1, -1, -1.1, -1.1, -1, 0, # minimum missile
        ],
        dtype=np.float32,
    ),
    high=np.array(
        [
            SCREEN_WIDTH + 10, SCREEN_HEIGHT + 10, MAX_DRONE_VX, MAX_DRONE_VY, MAX_DRONE_AX, MAX_DRONE_AY,  # agent states
            1000, 1000, 1.1, 1.1, np.linalg.norm([SCREEN_WIDTH + 10, SCREEN_HEIGHT + 10]), np.linalg.norm([SCREEN_WIDTH + 10, SCREEN_HEIGHT + 10]), # assigned missile
            1000, 1000, 1.1, 1.1, np.linalg.norm([SCREEN_WIDTH + 10, SCREEN_HEIGHT + 10]), np.linalg.norm([SCREEN_WIDTH + 10, SCREEN_HEIGHT + 10]), # minimum missile
        ],  
        dtype=np.float32,
    ),
    shape=(OBSERVATION_PER_DRONE * 1 + OBSERVATION_PER_MISSILE * 2,),
    dtype=np.float32,
)
ACTION_SPACE = gym.spaces.Box(low=-MAX_VELOCITY, high=MAX_VELOCITY, shape=(2,), dtype=np.int16)


class MissileDefenseEnv(MultiAgentEnv):
    def __init__(self, config=None, test_level=0, render=False, realistic_render=False):
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
        self.level = test_level
        self.selected_level = test_level
        self.missiles_data = {
            f"missile_{missile_id}": {
                "missile_pos": np.array([-1, -1], dtype=np.float32),
                "missile_velocity": np.zeros(2, dtype=np.float32),
                "missile_acceleration": np.random.uniform(0, MAX_MISSILE_ACCELERATION),
                "distance_to_base": MAX_DISTANCE_FROM_BASE,
                "neutralized": True,
                "cooldown": 0,  # Cooldown for missile
            }
            for missile_id in range(NUM_MISSILES)
        }
        for level in range(self.level + 1):
            self._increase_difficulty()
        self.step_counter = 0
        self.missiles_intercepted = 0
        self.survival_time = 0
        self._set_next_level_threshold()

    def reset(self, seed=None, options=None):
        self.survival_time = 0
        self.missiles_intercepted = 0
        global MAX_MISSILE_SPEED, MAX_MISSILE_ACCELERATION
        print("******reset*******")
        print(f"NUM MISSILE: {NUM_MISSILES}")
        print(f"Level: {self.level}, missile intercepted: {self.missiles_intercepted}, survival time: {self.survival_time}")
        print(f"Next Level Threshold: {self.next_level_threshold}")
        self.agents = [f"drone_{i}" for i in range(NUM_DRONES)]
        self.missiles = [f"missile_{i}" for i in range(NUM_MISSILES)]

        self.selected_level = self.level
        if random.random() > 0.8:
            self.selected_level = random.randint(0, self.level)
        print(f"Selected level: {self.selected_level}")

        # Initialize missile
        self.drone_assignment = {agent: None for agent in self.agents}
        self.closest_missile = None
        # Randomly select `NUM_MISSILES` missiles to be active
        active_missiles = random.sample(list(self.missiles_data.keys()), NUM_MISSILES)
        for missile_id in active_missiles:
            self._spawn_missile(missile_id)
        self._find_closest_missile()
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
        observations = {agent: self._get_obs(agent) for agent in self.agents}
        return observations, {}

    def _spawn_missile(self, missile_id):  # FOR RESPAWN
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
            "distance_to_base": np.linalg.norm([x, y]),
            "neutralized": False,
            "cooldown": 0,  # Cooldown for missile
        }

        assert not self.missiles_data[missile_id]["neutralized"], f"Missile {missile_id} failed to respawn"
        print(f"🚀 {missile_id} respawned at {self.missiles_data[missile_id]['missile_pos']}")

    def _find_closest_missile(self):
        closest_id = None
        closest_distance = float("inf")
        for missile_id, missile_data in self.missiles_data.items():
            distance = missile_data["distance_to_base"]
            if missile_data["neutralized"]:
                continue
            if distance < closest_distance:
                closest_distance = distance
                closest_id = copy.copy(missile_id)
        self.closest_missile = copy.copy(closest_id)

    def _calculate_rewards(self):
        # Compute rewards, dones, and infos
        rewards = {agent: STEP_PENALTY for agent in self.agents}
        dones = {agent: False for agent in self.agents}
        dones["__all__"] = False

        # Check for missile hitting base
        for missile_id, missile_data in self.missiles_data.items():
            if np.linalg.norm(missile_data["missile_pos"] - self.base_pos) < BASE_SIZE / 2:
                print("💥 Base hit, game lost and terminate")
                print(f"Total missile intercepted this episode: {self.missiles_intercepted}, survived {self.survival_time} timesteps")
                dones["__all__"] = True
                rewards = {agent: BASE_HIT_REWARD for agent in self.agents}
        
        # If drone went out of bounds, stop providing its stuffs
        for agent in list(self.agents):
            distance_from_base = np.linalg.norm(self.drones_pos[agent] - self.base_pos)
            if distance_from_base >= BOUND_RADIUS:
                rewards[agent] += NEAR_BOUND_PENALTY

            if np.any(self.drones_pos[agent] < 0) or np.any(self.drones_pos[agent] > SCREEN_WIDTH):
                rewards[agent] += OUT_OF_BOUND_REWARD
                print(f"💀 {agent} went out of bound at {self.drones_pos[agent]}, not providing any more observation")
                dones[agent] = True
                self.agents.remove(agent)
                print(self.drone_assignment)
                self.drone_assignment.pop(agent)

        # Check for drone intercepting missile, terminate
        for agent in list(self.agents):
            rewards[agent] += ALIVE_REWARD
            for missile_id, missile_data in self.missiles_data.items():
                if missile_data["neutralized"]:
                    continue
                missile_pos = missile_data["missile_pos"]
                direction = missile_pos - self.drones_pos[agent]
                distance = np.linalg.norm(direction)
                # Reward for interception
                if distance < MISSILE_SIZE + DRONE_SIZE:
                    self.missiles_intercepted += 1
                    rewards[agent] += DRONE_HIT_REWARD
                    missile_data["neutralized"] = True
                    missile_data["cooldown"] = MISSILE_COOLDOWN  # Start countdown
                    print(f"🎯 {agent} intercepted {missile_id}, respawning in {MISSILE_COOLDOWN} steps")
                    print(f"missiles intercepted: {self.missiles_intercepted}")

                    for assigned_agent, assignment in self.drone_assignment.items():
                        if assignment == missile_id:
                            self.drone_assignment[assigned_agent] = None
                    
            # Smoothness reward
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

        # Reward for actively pursue ASSIGNED or CLOSEST missiles
        for agent, assigned_missile in self.drone_assignment.items():
            agent_pos = self.drones_pos[agent]
            agent_vel = self.drones_vel[agent]
            agent_accel = self.drones_acc[agent]
            if not assigned_missile:
                continue
            assigned_missile_unit_vector, _ = self._calculate_missile_vectors(agent, assigned_missile)
            
            if missile_id == self.closest_missile:
                rewards[agent] += APPROACH_VEL_REWARD * np.dot(agent_vel, assigned_missile_unit_vector)
                rewards[agent] += APPROACH_ACC_REWARD * np.dot(agent_accel, assigned_missile_unit_vector)
            else:
                # Priority weight based on how close missile is to the base
                closest_missile_unit_vector, _ = self._calculate_missile_vectors(agent, self.closest_missile)
                assigned_threat = 1.0 - np.clip(self.missiles_data[assigned_missile]["distance_to_base"] / MAX_DISTANCE_FROM_BASE, 0, 1)
                closest_threat = 1.0 - np.clip(self.missiles_data[self.closest_missile]["distance_to_base"] / MAX_DISTANCE_FROM_BASE, 0, 1)
                scale = 5  # Tune this
                exp_assigned = np.exp(scale * assigned_threat)
                exp_closest = np.exp(scale * closest_threat)

                # Softmax weighting of reward for both side
                weight_sum = exp_assigned + exp_closest
                assigned_weight = exp_assigned / weight_sum
                closest_weight = exp_closest / weight_sum

                rewards[agent] += assigned_weight * APPROACH_VEL_REWARD * np.dot(agent_vel, assigned_missile_unit_vector)
                rewards[agent] += assigned_weight * APPROACH_ACC_REWARD * np.dot(agent_accel, assigned_missile_unit_vector)
                rewards[agent] += closest_weight * APPROACH_VEL_REWARD * np.dot(agent_vel, closest_missile_unit_vector)
                rewards[agent] += closest_weight * APPROACH_ACC_REWARD * np.dot(agent_accel, closest_missile_unit_vector)
                # blended_dir = assigned_weight * assigned_dir_unit + closest_weight * closest_dir_unit
                # blended_dir /= np.linalg.norm(blended_dir) + 1e-8

                # rewards[agent] += APPROACH_VEL_REWARD * np.dot(agent_vel, blended_dir)
                # rewards[agent] += APPROACH_ACC_REWARD * np.dot(agent_accel, blended_dir)

        if len(self.agents) == 0:
            print("💀💀💀******No agent left, terminate******💀💀💀")
            dones["__all__"] = True
        return rewards, dones
    
    def _calculate_missile_vectors(self, agent_id, missile_id):
        agent_pos = self.drones_pos[agent_id]
        missile_pos = self.missiles_data[missile_id]["missile_pos"]
        direction = missile_pos - agent_pos
        distance = np.linalg.norm(direction)
        unit_vector = direction / distance if distance > 0 else np.zeros(2) # Normalize direction vector
        return unit_vector, distance

    def step(self, actions):
        self.survival_time += 1
        self.step_counter += 1
        respawned_missiles = []
        if (self.missiles_intercepted >= self.next_level_threshold["interceptions_needed"]
            and self.survival_time >= self.next_level_threshold["survival_steps"]):
            print(f"Level: {self.level}, missile intercepted: {self.missiles_intercepted}, survival time: {self.survival_time}")
            self.level += 1
            print(f"⏫ Level up to {self.level}!")
            self._increase_difficulty()
            print(f"Next Level Threshold: {self.next_level_threshold}")

        for missile_id, missile_data in self.missiles_data.items():
            if missile_data["neutralized"]:
                # For neutralized missiles, update cooldown and possibly respawn.
                if missile_data.get("cooldown", 0) > 0:
                    missile_data["cooldown"] -= 1
                elif missile_data.get("cooldown", 0) == 0:
                    self._spawn_missile(missile_id)
                    respawned_missiles.append(missile_id)
        # if respawned_missiles:
        #     print(f"Respawned missiles: {respawned_missiles} at step{self.step_counter}")

        # Update missiles positions
        minimum_distance = float("inf")
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
                    if (
                        speed > MAX_MISSILE_SPEED
                    ):  # --> Change these instances to global to default and remove keyboard input
                        missile_velocity = missile_velocity / speed * MAX_MISSILE_SPEED
                # Update missile position based on its velocity
                missile_pos += missile_velocity
                self.missiles_data[missile_id]["missile_pos"] = missile_pos
                self.missiles_data[missile_id]["missile_velocity"] = missile_velocity
                self.missiles_data[missile_id]["distance_to_base"] = distance

        # Update stored states
        for agent in self.agents:
            self.prev_drones_pos[agent] = np.copy(self.drones_pos[agent])
            self.prev_drones_vel[agent] = np.copy(self.drones_vel[agent])
            self.prev_drones_acc[agent] = np.copy(self.drones_acc[agent])
            
        # Update drones positions and velocities
        for agent, action in actions.items():
            if agent in self.drones_pos:
                desired_v = np.clip(action, -MAX_VELOCITY, MAX_VELOCITY)
                delta_v = desired_v - self.drones_vel[agent]
                max_delta_v = MAX_ACCELERATION * TIME_STEP
                if np.linalg.norm(delta_v) > max_delta_v:
                    delta_v = delta_v / np.linalg.norm(delta_v) * max_delta_v
                self.drones_vel[agent] += delta_v
        for agent in self.agents:
            self.drones_pos[agent] += self.drones_vel[agent] * TIME_STEP

        # Update acceleration
        for agent in self.agents:
            self.drones_acc[agent] = (self.drones_vel[agent] - self.prev_drones_vel[agent]) / TIME_STEP

        self._find_closest_missile()
        rewards, dones = self._calculate_rewards()
        self._assign_missiles()

        # update the observations, dont include the drones that went out of bound, else the training will crash
        observations = {agent: self._get_obs(agent) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        # we dont need this truncateds as the episode will terminate when base is hit
        truncateds = {agent: False for agent in self.agents}
        truncateds["__all__"] = False

        if self.render_flag:
            self.render()
        return observations, rewards, dones, truncateds, infos

    def _set_next_level_threshold(self):
        self.next_level_threshold = {"interceptions_needed": self.level*NUM_DRONES, "survival_steps": self.level*LEVEL_SURVIVAL_MULTIPLIER}
        # self.survival_time = 0
        # self.missiles_intercepted = 0
        # Resetting seems to slow down learning

    def _increase_difficulty(self):  # increase the difficulty of the game as per new parameters
        global MAX_MISSILE_SPEED, MAX_MISSILE_ACCELERATION, MISSILE_COOLDOWN, NUM_MISSILES
        previous_num_missile = copy.copy(NUM_MISSILES)
        self._set_next_level_threshold()
        MAX_MISSILE_ACCELERATION *= LEVEL_ACC_MULTIPLIER
        MAX_MISSILE_SPEED *= LEVEL_VEL_MULTIPLIER
        MISSILE_COOLDOWN = max(0, MISSILE_COOLDOWN - 10)  # Decrease cooldown
        NUM_MISSILES = int(NUM_DRONES * LEVEL_MISSILE_MULTIPLIER**self.level)
        print(f"Max Missile Acceleration: {MAX_MISSILE_ACCELERATION}")
        print(f"Max Missile Speed: {MAX_MISSILE_SPEED}")
        print(f"Missile Number: {NUM_MISSILES}")
        print(f"Missile Cooldown: {MISSILE_COOLDOWN}")
        print(f"Next level threshold: {self.next_level_threshold}")
        self.missiles = [f"missile_{i}" for i in range(NUM_MISSILES)]
        if NUM_MISSILES != previous_num_missile:
            num_diff = NUM_MISSILES - previous_num_missile
            for new_missile_id in range(num_diff):
                print(f"Adding new missile: {previous_num_missile + new_missile_id}")
                self._spawn_missile(f"missile_{previous_num_missile + new_missile_id}")

    def _assign_missiles(self, distance_threshold=RETARGET_DISTANCE):
        """
        Assigns missiles to drones based on the following strategy:

        1. If more than half the drones are unassigned, reassign all using a greedy heap approach.
        2. Otherwise, only assign new missiles to currently unassigned drones.
        """
        previous_assignments = self.drone_assignment.copy()
        # Compute distances for all drones to all active missiles
        agent_missile_distances = {agent: {} for agent in self.agents}
        for missile_id, missile_data in self.missiles_data.items():
            if missile_data["neutralized"]:
                continue  # Skip neutralized missiles
            for agent in self.agents:
                agent_missile_distances[agent][missile_id] = np.linalg.norm(
                    self.drones_pos[agent] - missile_data["missile_pos"]
                )
        for agent in self.agents:
            if agent not in agent_missile_distances.keys():
                agent_missile_distances[agent] = {}
        # Check if reassignment would improve more than half the drones
        drones_with_better_options = 0
        distance_heap = []
        for agent, missile_distances in agent_missile_distances.items():
            for missile, distance in missile_distances.items():
                heapq.heappush(distance_heap, (distance, agent, missile))

        # Evaluate if any assigned drone could have a significantly closer target
        for agent, current_missile in self.drone_assignment.items():
            if current_missile is None:  # Current agent has no target
                continue
            current_distance = agent_missile_distances[agent].get(current_missile, float("inf"))
            for solution in distance_heap:
                if solution[1] != agent or (solution[1] == agent and solution[2] == current_missile):
                    continue
                best_distance, _, best_missile = solution
                if abs(current_distance - best_distance) > distance_threshold:
                    drones_with_better_options += 1

        # If more than half the drones are unassigned, reset and use heapq to reassign all
        unassigned_drones = {agent for agent in self.agents if self.drone_assignment.get(agent) is None}
        num_unassigned_drones = len(unassigned_drones)
        if num_unassigned_drones > len(self.agents) / 2 or drones_with_better_options > len(self.agents) / 2:
            assigned_drones = {}
            assigned_missiles = set()
            while distance_heap:
                distance, agent, missile = heapq.heappop(distance_heap)
                if agent in assigned_drones or missile in assigned_missiles:
                    continue
                assigned_drones[agent] = missile
                assigned_missiles.add(missile)
            for agent in self.agents:
                if agent not in assigned_drones:
                    assigned_drones[agent] = None
            assigned_drones = {k: assigned_drones[k] for k in sorted(assigned_drones)}
            self.drone_assignment = assigned_drones
            return

        # Otherwise, only assign unassigned drones to unassigned missiles
        unassigned_missiles = {
            missile for missile in self.missiles_data if missile not in self.drone_assignment.values()
        }
        if not unassigned_missiles or not unassigned_drones:
            return  # No assignments needed

        # Create a heap with only unassigned drones and unassigned missiles
        distance_heap = []
        for agent in unassigned_drones:
            for missile in unassigned_missiles:
                if missile in agent_missile_distances[agent]:  # Check if missile is valid
                    heapq.heappush(distance_heap, (agent_missile_distances[agent][missile], agent, missile))

        # Assign remaining missiles to remaining drones
        while distance_heap:
            distance, agent, missile = heapq.heappop(distance_heap)
            if agent not in self.drone_assignment and missile not in self.drone_assignment.values():
                self.drone_assignment[agent] = missile  # Assign missile to the agent

        # if self.drone_assignment != previous_assignments:
        #     print(f"Updated Assignments: {self.drone_assignment}")
        
        # Validate assignments
        assigned_list = list(self.drone_assignment.values())
        for missile_id in self.missiles:
            count = assigned_list.count(missile_id)
            if count > 1:
                print(f"❗WARNING: {missile_id} is assigned to {count} drones")

    def _get_obs(self, agent):
        obs = [self.drones_pos[agent][0], self.drones_pos[agent][1]]
        obs.extend(self.drones_vel[agent].tolist())  # vx, vy
        obs.extend(self.drones_acc[agent].tolist())  # ax, ay

        # Provide data of assigned missile to agent
        if agent in self.drone_assignment and self.drone_assignment[agent] is not None:
            missile_data = self.missiles_data[self.drone_assignment[agent]]
            assigned_missile_unit_vector, assigned_missile_distance_to_agent = self._calculate_missile_vectors(agent, self.drone_assignment[agent])
            obs.extend(missile_data["missile_pos"])
            obs.extend(assigned_missile_unit_vector.tolist())
            obs.extend([assigned_missile_distance_to_agent])
            obs.extend([missile_data["distance_to_base"]])
        else:
            obs.extend([-1, -1, 0, 0, -1, 999])
        # print(obs)

        # Provide data of closest missile to agent
        if self.closest_missile is not None:
            closest_missile_data = self.missiles_data[self.closest_missile]
            closest_missile_unit_vector, closest_missile_distance_to_agent = self._calculate_missile_vectors(agent, self.closest_missile)
            obs.extend(closest_missile_data["missile_pos"])
            obs.extend(closest_missile_unit_vector.tolist())
            obs.extend([closest_missile_distance_to_agent])
            obs.extend([closest_missile_data["distance_to_base"]])
        else:
            obs.extend([-1, -1, 0, 0, -1, 999])
        # print(obs)
        # input()
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
            font = pygame.font.Font(None, 35)  # Choose a suitable font and size

            # Reverse the allocation dictionary to map missiles to drones
            missile_to_drone = {v: k for k, v in self.drone_assignment.items()}

            for missile_id, missile_data in self.missiles_data.items():

                missile_direction = missile_data["missile_velocity"]
                angle = -np.degrees(np.arctan2(missile_direction[1], missile_direction[0]))
                rotated_missile_img = pygame.transform.rotate(self.missile_png, angle)
                missile_rect = rotated_missile_img.get_rect(center=missile_data["missile_pos"].astype(int))
                self.screen.blit(rotated_missile_img, missile_rect)

                # Check if this missile is allocated to a drone
                if missile_id in missile_to_drone:
                    drone_id = missile_to_drone[missile_id]  # Get the assigned drone
                    drone_number = int(drone_id.split("_")[-1])  # Extract drone number

                    # Render the text near the missile
                    text_surface = font.render(f"{drone_number}", True, (0, 0, 0))  # black text
                    text_rect = text_surface.get_rect(
                        center=(missile_data["missile_pos"][0] + 15, missile_data["missile_pos"][1] - 15)
                    )
                    self.screen.blit(text_surface, text_rect)

            for i, drone_pos in enumerate(self.drones_pos.values()):
                drone_rect = self.drone_png.get_rect(center=drone_pos.astype(int))
                self.screen.blit(self.drone_png, drone_rect)
                # Render the text (drone number) slightly offset from the drone position
                text_surface = font.render(f"{i}", True, (0, 0, 0))  # black text
                text_rect = text_surface.get_rect(center=(drone_pos[0] + 15, drone_pos[1] - 15))
                self.screen.blit(text_surface, text_rect)

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
    env = MissileDefenseEnv(render=False, realistic_render=True)
    while True:
        obs = env.reset()
        print(env.drone_assignment)
        done = False
        # actions = {"drone_0": np.array([0.1, 0.1]), "drone_1": np.array([-0.1, -0.1]), "drone_2": np.array([0.0, 0.0])}
        # obs, rewards, terminateds, truncateds, infos = env.step(actions)
        while not done:
            # actions = {"drone_0": np.array([0.1, 0.1]), "drone_1": np.array([-0.1, -0.1]), "drone_2": np.array([0.0, 0.0])}
            actions = {
                "drone_0": np.array([1.0, 0.0]),
                "drone_1": np.array([-0.0, -0.0]),
                "drone_2": np.array([0.0, 0.0]),
            }
            obs, rewards, dones, truncateds, infos = env.step(actions)
            done = dones["__all__"]
    env.close()
