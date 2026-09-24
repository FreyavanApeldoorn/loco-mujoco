import numpy as np
import pickle
import jax
import jax.numpy as jnp
import opensim as osim

from loco_mujoco import TaskFactory, LocoEnv
from loco_mujoco.task_factories import ImitationFactory, LAFAN1DatasetConf, DefaultDatasetConf, AMASSDatasetConf
from loco_mujoco.algorithms import PPOJax
from loco_mujoco.algorithms.ppo_jax import PPOAgentConf, PPOAgentState
from opensim import Model

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

def adjust_policy_fn_for_OpenSim(policy: Callable, osim_model: Model, remove_keywords: list) -> Callable:
    '''
    Creates a controller that takes an OpenSim state as an input and outputs ..., based on a previously trained policy.

    Inputs:
    policy: trained RL policy (see create_policy_fn)
    osim_model: the OpenSim model that the controller will be applied to
    remove_keywords: a list of keywords that indicate which states need to be removed between the opensim state and the mujoco observation

    Outputs:
    OpenSim_controller: controller that takes the opensim state as an input and outputs muscle actions
    
    '''
    # Remove from the Opensim state:
    # all muscle activations, all fiber lengths, pelvis_tx, subtalar angle, mtp angle

    def OpenSim_controller(osim_state):
        state_vector = osim_model.getStateVariableValues(osim_state)
        state_names = osim_model.getStateVariableNames()

        all_state_names = [state_names.get(i) for i in range(state_names.getSize())]

        # This is probably objectively the slowest way to do this, should rewrite this at some point
        obs = []
        state_names_removed = []
        for i in range(len(all_state_names)):
            if not any([r in all_state_names[i] for r in remove_keywords]):
                obs.append(state_vector[i])
                state_names_removed.append(all_state_names[i])

        all_angles = [obs[i] for i in range(len(state_names_removed)) if 'value' in state_names_removed[i]]
        all_speeds = [obs[i] for i in range(len(state_names_removed)) if 'speed' in state_names_removed[i]]

        obs = all_angles + all_speeds

        action = policy(obs)

        action = action[14:] #Removes the last 14 actuators for the arms, this should probably be a variable...

        return action

    return OpenSim_controller

@hydra.main(version_base=None, config_path="./", config_name='conf')
def create_RL_controller(config: DictConfig) -> None:
    '''
    Inputs:
    config -> The configurations in the conf.yaml file

    Trains a RL controller based on the configurations in the conf.yaml file
    '''

    # print(OmegaConf.to_yaml(config, resolve=False))
    # OmegaConf.resolve(config)

    #n_substeps specifies the number of simulation steps executed for each call to the environment's step() function.
    env = ImitationFactory.make(config.experiment.env_params.env_name,
                                default_dataset_conf=DefaultDatasetConf([config.experiment.env_params.default_dataset]),
                                # lafan1_dataset_conf=LAFAN1DatasetConf([config.experiment.env_params.lafan_dataset]), # This doesn't work
                                n_substeps=config.experiment.env_params.n_substeps
                                )

    print('control_dt: ', env.dt)

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


def OpenSim_RL_implementation(osim_model: Model, state_to_action: Callable, control_dt, duration) -> None:
    '''
    Implements the RL controller in OpenSim and renders a video of it
    
    Inputs:
    osim_model: The OpenSim Model
    state_to_action: RL policy that takes the opensim state as an input and outputs a vector with muscle actions
    muscle_names: list of muscle names of the osim model
    control_dt: timestep
    duration: length of simulation in seconds
    '''

    muscle_names = [i.getName() for i in osim_model.getMuscles()]

    controller = osim.PrescribedController()

    muscles = osim_model.getMuscles()
    for name in muscle_names:
        controller.addActuator(muscles.get(name))
        controller.prescribeControlForActuator(name, osim.Constant(0.0))

    osim_model.addController(controller)

    osim_model.setUseVisualizer(True)

    state = osim_model.initSystem()

    # osim_model.equilibrateMuscles(state)

    visualizer = osim_model.getVisualizer()
    visualizer.show(state)

    functions = controller.upd_ControlFunctions()
    controls = [osim.Constant.safeDownCast(functions.get(i)) for i in range(len(muscle_names))]

    end_time = state.getTime() + duration

    while state.getTime() < end_time:
        action = np.asarray(state_to_action(state), dtype=np.float64).reshape(-1)

        for function, excitation in zip(controls, action):
            function.setValue(float(excitation))

        state.updY()

        next_time = min(state.getTime() + control_dt, end_time)

        manager = osim.Manager(osim_model)
        manager.setIntegratorAccuracy(1e-5)
        manager.initialize(state)
        state = osim.State(manager.integrate(next_time))

        visualizer.show(state)

    return

if __name__=='__main__':
    save_path_RL = 'working_folder/saved_agent' #This should match the one in conf.yaml
    save_path_osim_model = 'working_folder/models/Body_model_opensim_added_floor.osim'

    TRAINING = True #re-trains the RL controller

    #Enable tests here
    POLICY_TEST = False
    MODEL_MATCHING_TEST = False

    if TRAINING:
        create_RL_controller()

    if os.path.isdir(save_path_RL):
        agent_conf, agent_state = PPOJax.load_agent('working_folder/saved_agent'+'/PPOJax_saved.pkl')
    else:
        raise Exception('The file path does not point to a loaded agent :(')


    policy = create_policy_fn(agent_state, agent_conf)

    osim_model = osim.Model(save_path_osim_model)

    if POLICY_TEST or MODEL_MATCHING_TEST:
        env = ImitationFactory.make('MjxSkeletonMuscle',
                                default_dataset_conf=DefaultDatasetConf(["walk"]),
                                n_substeps=20)
        # PPOJax.play_policy(env, agent_conf, agent_state, deterministic=True, n_steps=100, n_envs=1, record=False,
        #             train_state_seed=0) 

    remove_keywords = ['activation', 'fiber', 'pelvis_tx/value', 'subtalar_angle', 'mtp_angle']

    state_to_action = adjust_policy_fn_for_OpenSim(policy, osim_model, remove_keywords)

    OpenSim_RL_implementation(osim_model, state_to_action, 0.04, 1.0) #control_dt based on env.dt in the RL training loop 

    

# --------------Policy test ----------------------------------------------------------------------------------------------------------
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

# -------------- Model matching test ------------------------------------------------------------------------------------------------
    if MODEL_MATCHING_TEST:

        loco_actuators = [env._model.actuator(i).name for i in range(env._model.nu)]
        osim_actuators = [i.getName() for i in osim_model.getMuscles()]

        loco_actuators = loco_actuators[14:]

        for i in range(len(loco_actuators)):
            print(f'index: {i}, loco name: {loco_actuators[i]}')

        print('OPENSIM ONES')

        for i in range(len(osim_actuators)):
            print(f'index: {i}, osim name:{osim_actuators[i]}')

        print(osim_actuators == loco_actuators)


        rng = jax.random.PRNGKey(0)
        loco_obs = env.reset() #Gets the initial observation
        osim_state = osim_model.initSystem()


        osim_state = osim_model.initSystem()
        state_names = osim_model.getStateVariableNames()

        all_states = [state_names.get(i) for i in range(state_names.getSize())]

        osim_obs = []
        for i in range(len(all_states)):
            if not any([r in all_states[i] for r in remove_keywords]):
                osim_obs.append(all_states[i])

        all_angles = [a for a in osim_obs if 'value' in a]
        all_speeds = [s for s in osim_obs if 'speed' in s]

        osim_obs = all_angles + all_speeds

        print('Visual Inpection: check that the parameter names match')

        for o in range(len(osim_obs)):
            print(f"index: {o}, osim: {osim_obs[o]}")

        for o in env.obs_container.values():
            print(f"index {o.obs_ind}, loco {o.name}")

        print('PASSED MODEL MATCHING TEST :)')


