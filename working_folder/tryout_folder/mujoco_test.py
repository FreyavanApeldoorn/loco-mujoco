import numpy as np
import jax
import jax.numpy as jnp

from loco_mujoco import TaskFactory
from loco_mujoco.task_factories import ImitationFactory, LAFAN1DatasetConf, DefaultDatasetConf, AMASSDatasetConf
from loco_mujoco.algorithms import PPOJax

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import traceback

@hydra.main(version_base=None, config_path="./", config_name='conf_test')
def experiment(config: DictConfig):
    
    # I should get this from the config file but I'm not sure how
    env = ImitationFactory.make(#'MjxSkeletonMuscle',
                                "MjxUnitreeH1",
                                default_dataset_conf=DefaultDatasetConf(["walk"]),
                                n_substeps=20)

    if config.experiment.visualize:
        env.play_trajectory(n_episodes=1, n_steps_per_episode=50, render=True)

    #initial agent configuration
    agent_conf = PPOJax.init_agent_conf(env, config)

    #training function
    train_fn = PPOJax.build_train_fn(env, agent_conf)
    train_fn = jax.jit(train_fn) 

    rng = jax.random.PRNGKey(0)
    out = train_fn(rng) #Returns agent_State, training_metrics and validation_metrics

    agent_state = out["agent_state"] #This is basically a wrapped jax.training.train_state

    if config.experiment.visualize:
        PPOJax.play_policy(env, agent_conf, agent_state, deterministic=True, n_steps=100, n_envs=1, record=False,
                            train_state_seed=0)

    return agent_state

if __name__ == '__main__':
    agent_state = experiment()



# if CREATE_ENVIRONMENT:
#     env = ImitationFactory.make('MjxSkeletonMuscle',
#                                 default_dataset_conf=DefaultDatasetConf(["walk"]),
#                                 n_substeps=20)
    
#     with open('env', 'ab') as envfile:
#         pickle.dump(env, envfile)
# else:
#     with open('env', 'rb') as envfile:
#         try:
#             while True:
#                 env = pickle.load(envfile)
#         except EOFError:
#             pass



    

    


    