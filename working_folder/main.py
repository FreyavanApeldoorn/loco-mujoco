import numpy as np
import pickle
import jax
import jax.numpy as jnp
import opensim as osim

from loco_mujoco import TaskFactory
from loco_mujoco.task_factories import ImitationFactory, LAFAN1DatasetConf, DefaultDatasetConf, AMASSDatasetConf
from loco_mujoco.algorithms import PPOJax
from loco_mujoco.algorithms.ppo_jax import PPOAgentConf, PPOAgentState

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import traceback
import typing
from collections.abc import Callable
import os

def create_policy_fn(agent_state: PPOAgentState, agent_conf: PPOAgentConf) -> Callable: 
    '''
    Creates a policy function that takes an observation as an input and returns an action.

    Inputs:
    agent_state: PPOAgentState object from create_RL_controller
    agent_conf: PPOAgentState object from create_RL_controller
    
    Outputs:
    policy: Policy function
    '''

    def policy(obs):
        (pi, _), _ = agent_conf.network.apply(
            {
                "params": agent_state.train_state.params,
                "run_stats": agent_state.train_state.run_stats
            },
            obs,
            mutable=['run_stats']
        )

        action = pi.mean()

        return action

    return policy

@hydra.main(version_base=None, config_path="./", config_name='conf')
def create_RL_controller(config: DictConfig) -> None:
    '''
    Inputs:
    config -> The configurations in the conf.yaml file

    Trains a RL controller based on the configurations in the conf.yaml file
    '''

    #n_substeps specifies the number of simulation steps executed for each call to the environment's step() function.
    env = ImitationFactory.make(config.experiment.env_params.env_name,
                                default_dataset_conf=DefaultDatasetConf([config.experiment.env_params.default_dataset]),
                                # lafan1_dataset_conf=LAFAN1DatasetConf([config.experiment.env_params.lafan_dataset]), # This doesn't work
                                n_substeps=config.experiment.env_params.n_substeps
                                )

    if config.experiment.visualize:
        env.play_trajectory(n_episodes=1, n_steps_per_episode=100, render=True)

    #initial agent configuration
    agent_conf = PPOJax.init_agent_conf(env, config)

    #training function
    train_fn = PPOJax.build_train_fn(env, agent_conf)
    train_fn = jax.jit(train_fn) 

    rng = jax.random.PRNGKey(0)
    out = train_fn(rng) #Returns agent_state, training_metrics and validation_metrics

    agent_state = out["agent_state"] #This is basically a wrapped jax.training.train_state

    if config.experiment.visualize:
        PPOJax.play_policy(env, agent_conf, agent_state, deterministic=True, n_steps=100, n_envs=1, record=False,
                            train_state_seed=0) 
      
    os.makedirs(config.experiment.save_path, exist_ok=True)
    save_path = PPOJax.save_agent(config.experiment.save_path, agent_conf, agent_state)

def OpenSim_RL_retraining():
    '''
    Inputs:
    agent_state -> Trained RL policy

    Outputs:
    agent_state_adjusted ->

    Adjusts the RL controller trained in create_RL_controller for OpenSim, maintains controller structure
    '''
    return

def OpenSim_RL_conversion(policy: Callable):
    '''
    Inputs:

    Outputs:

    Converts the MuJoCo controller into a form that can be implemented in OpenSim
    '''

    #The OpenSim model only has the muscles as actuators, not the upper-body ones. 


    return

def OpenSim_RL_implementation(model_path: str):
    '''
    Inputs:
    model_path -> Path to where the OpenSim model is located

    Outputs:

    Applies the RL controller in OpenSim environment
    '''
    osim_model = osim.Model(model_path)
    actuators = osim_model.getActuators()


    return

if __name__=='__main__':
    save_path_RL = 'working_folder/saved_agent' #This should match the one in conf.yaml
    save_path_osim_model = 'working_folder/models/Body_model_opensim.osim'

    TRAINING = False #re-trains the RL controller

    #Enable tests here
    POLICY_TEST = False
    MODEL_MATCHING_TEST = True

    if TRAINING:
        create_RL_controller()

    if os.path.isdir(save_path_RL):
        agent_conf, agent_state = PPOJax.load_agent('working_folder/saved_agent'+'/PPOJax_saved.pkl')
    else:
        raise Exception('The file path does not point to a loaded agent :(')


    policy = create_policy_fn(agent_state, agent_conf)

    osim_model = osim.Model(save_path_osim_model)

    #env.step()

    if POLICY_TEST or MODEL_MATCHING_TEST:
        env = ImitationFactory.make('MjxSkeletonMuscle',
                                default_dataset_conf=DefaultDatasetConf(["walk"]),
                                n_substeps=20)

    # --------------Policy test ---------------------------------------------------------
    if POLICY_TEST:
        obs = env.reset() # numpy.ndarray (65,) 
        # 5 root positions, 27 joint positions, 6 root velocities, 27 joint velocities

        action = policy(obs) #jaxlib._jax.ArrayImpl (106,)
        # 14 upper body torque actuators, 92 muscle actuators

        if len(obs) != 65: #maybe replace this with model parameters instead of just an integer?
            raise Exception('The policy input is not the correct shape :(')
        elif len(action)!= len([env._model.actuator(i).name for i in range(env._model.nu)]):
            raise Exception('The policy output is not the correct shape :(')

        print('PASSED POLICY TEST :)')

    # -------------- Model matching test --------------------------
    if MODEL_MATCHING_TEST:

        loco_actuators = [env._model.actuator(i).name for i in range(env._model.nu)]
        osim_actuators = [i.getName() for i in osim_model.getActuators()]

        rng = jax.random.PRNGKey(0)
        loco_observation = env.mjx_reset(rng) #Gets the initial state
        osim_state = osim_model.initSystem()

        print(loco_observation)

        osim_state = osim_model.initSystem()
        state_names = osim_model.getStateVariableNames()

        all_states = [state_names.get(i) for i in range(state_names.getSize())]
        joint_values = [j for j in all_states if 'value' in j]
        joint_speeds = [j for j in all_states if 'speed' in j]
        muscle_activations = [j for j in all_states if 'activation' in j]
        muscle_fiber_lengths = [j for j in all_states if 'fiber' in j]


        print('PASSED MODEL MATCHING TEST :)')


