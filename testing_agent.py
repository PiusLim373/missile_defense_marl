import gymnasium as gym
from ray.rllib.algorithms.ppo import PPOConfig, PPO
from ray.tune.registry import register_env
from missile_defense_gym import *
import os
import pygame

# Register the custom environment
env_name = "missile_defense_env"
register_env(env_name, lambda config: MissileDefenseEnv(config, render=False))

# Define the checkpoint path (update this to your actual checkpoint location)
pwd = os.getcwd()
model_path = input("Enter your model path: \n")
checkpoint_path = os.path.join(pwd, model_path if model_path else "sample_trained_model/checkpoint_000000")

# Load the trained model
config = (
    PPOConfig()
    .environment(env=env_name)
    .framework("torch")
    .rollouts(num_rollout_workers=0)  # No workers needed for testing
)

algo = PPO.from_checkpoint(checkpoint_path)

# Create the environment for testing
env = MissileDefenseEnv(render=True, realistic_render=True, test_level=3)

while True:
    # Run a test episode
    observations, _ = env.reset()
    done = False

    while not done:
        for event in pygame.event.get():    #Added to change missile speed, comment it out if not needed, change self.MAX_MISSILE_SPEED to MAX_MISSILE_SPEED in missile_defense_gym.py
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    env.MAX_MISSILE_SPEED *= 1.1
                    print("Increased missile speed")
                elif event.key == pygame.K_DOWN:
                    env.MAX_MISSILE_SPEED *= 0.9
                    print("Decreased missile speed")
                elif event.key == pygame.K_r:
                    env.MAX_MISSILE_SPEED = 2.0
                    print("Reset missile speed to default")
                    
                






        actions = {}
        for agent_id, obs in observations.items():
            actions[agent_id] = algo.compute_single_action(
                obs, policy_id="shared_policy")

        observations, rewards, dones, truncateds, infos = env.step(actions)
        done = dones["__all__"]

env.close()
