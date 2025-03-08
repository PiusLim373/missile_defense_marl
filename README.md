# missile_defense_marl
This is a multi agent machine learning based missile defense system project for NUS MSc in Robotics module ME5424: Swarm Robotics and Aerial Robotics.

This project aims to deploy a fleet of drones that will intercept and defense a base through machine learning 

## Setup
### 1. Clone the repo
```
git clone https://github.com/PiusLim373/missile_defense_marl.git
```
### 2. Setup using Conda
```
cd missile_defense_marl
conda env create -f requirement.yml
conda activate me5424-missile-defense-marl
```
This will create a conda environment me5424-missile-defense-marl with necessary packages installed to run the project.

## Training with RLLIB MAPPO Algorithm
```
python training_agent.py
```
:warning: User can opt to render the pygame visualizer during training, but the training time will be significantly slower

![](/docs/pygame_visualizer.png)

After the training completed, a `PPO_<date>` will be saved to the `models/` directory. The `models/PPO_<date>/checkpoint_000xxx/` is the actual model that can be loaded.

## Testing with Trained Model
```
python testing_agent.py
```
A sample model is provided, feel free to change it.

![](/docs/sample_trained_agent_playouts.mp4)