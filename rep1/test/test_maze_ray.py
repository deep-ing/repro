
import ray 
from ray import tune
from ray.tune.registry import register_env
import os 
import sys 

PROJECT_PATH = os.getcwd()
SOURCE_PATH = os.path.join(
    PROJECT_PATH,"src"
)
sys.path.append(SOURCE_PATH)


env_config = {}

experiment_config = {
    "agent" : "PPO",
    "num_workers" : 2,
    "num_gups" : 1,
    "local_dir" : "results",
    "checkpoint_freq" : 5,
    "checkpoint_at_end" : True,
    "checkpoint_name" : "test_name",
    "trial_dirname_creator" : "trial",
    "verbose" : 1,
    "stop":{
        "episode_reward_mean": 1e9,
        "training_iteration" : 1000
    },
    "restore" : False
}


from maze import MazeEnv
register_env('maze_env', lambda config : MazeEnv())

train_config = {
    "framework" : "torch",
    "env" : "maze_env"
}

tune.run("PPO",
        config=train_config,
        stop=experiment_config['stop'],
        checkpoint_freq=experiment_config['checkpoint_freq'],
        checkpoint_at_end=experiment_config['checkpoint_at_end'],
        local_dir=experiment_config['local_dir'],
        name=experiment_config['checkpoint_name'],
        restore=None,
        verbose=1)