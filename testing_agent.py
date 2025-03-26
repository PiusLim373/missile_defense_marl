import gymnasium as gym
from ray.rllib.algorithms.ppo import PPOConfig, PPO
from ray.tune.registry import register_env
from missile_defense_gym import *
import os

# Register the custom environment
env_name = "missile_defense_env"
register_env(env_name, lambda config: MissileDefenseEnv(config, render=False))

# Define the checkpoint path (update this to your actual checkpoint location)
pwd = os.getcwd()
model_path = input("Enter your model path")
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
        actions = {}
        for agent_id, obs in observations.items():
            actions[agent_id] = algo.compute_single_action(
                obs, policy_id="shared_policy")

        observations, rewards, dones, truncateds, infos = env.step(actions)
        done = dones["__all__"]
        # print(f"Actions: {actions}")
        # print(f"Rewards: {rewards}")
        # print(f"Dones: {dones}")
        # print(len(observations))
        # for key,value in observations.items():
        #     print(f"{key}: {value}")
        # for key, value in observations.items():
        #     if value.shape[0] != 33:        #         print("error!")
            # print(f"{key}: {value.shape}")  # or len(value) 

env.close()
