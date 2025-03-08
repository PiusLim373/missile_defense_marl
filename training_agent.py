import gymnasium as gym
from ray import air, tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from missile_defense_gym import *
import os 

cwd = os.getcwd()
checkpoint_path = os.path.join(cwd, "models")

# Register the custom environment
env_name = "missile_defense_env"
register_env(env_name, lambda config: MissileDefenseEnv(config, render=False))


# Configure MAPPO with shared policies
config = (
    PPOConfig()
    .environment(env=env_name)
    .rollouts(num_rollout_workers=1)
    .training(
        lr=5e-4,  # Learning rate
        gamma=0.99,  # Discount factor
        lambda_=0.95,  # GAE parameter
        use_gae=True,
        vf_loss_coeff=1.0,
        entropy_coeff=0.01,
        train_batch_size=4000,
        sgd_minibatch_size=256,
        num_sgd_iter=10,
        clip_param=0.2,
    )
    .multi_agent(
        policies={"shared_policy": (
            None, OBSERVATION_SPACE, ACTION_SPACE, {})},
        policy_mapping_fn=lambda agent_id, episode, **kwargs: "shared_policy",
        policies_to_train=["shared_policy"]
    )
    .framework("torch")
)

# Training with Tune
tune.Tuner(
    "PPO",
    param_space=config.to_dict(),
    run_config=air.RunConfig(
        stop={"episodes_total": 2000},
        storage_path=checkpoint_path,  # Specify the custom save directory
        checkpoint_config=air.CheckpointConfig(
            checkpoint_frequency=3,  # Save a checkpoint every 10 training iterations
            checkpoint_at_end=True,   # Also save at the end of training
        ),
    ),
).fit()
