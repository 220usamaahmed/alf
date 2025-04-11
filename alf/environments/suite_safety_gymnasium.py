try:
    import mujoco_py
    import safety_gymnasium    
except ImportError:
    mujoco_py = None
    safety_gymnasium = None

import gym
import gymnasium
from typing import Callable, List
import numpy as np
import copy

import alf
from alf.environments import suite_gym
from torch.utils.tensorboard.writer import SummaryWriter

def is_available():
    """Check if both ``mujoco_py`` and ``safety_gym`` have been installed."""
    return (mujoco_py is not None and safety_gymnasium is not None)


TESTING = 1

class GymnasiumWrapper(gym.Wrapper):
    def __init__(self, env):

        def seed(seed):
            ...

        super().__init__(env)

        self.seed = seed
        self.num_steps = 1000
        
        self.observation_space = self._convert_gymnasium_to_gym_box(self.observation_space)
        self.action_space = self._convert_gymnasium_to_gym_box(self.action_space)

        self._reward_space = gym.spaces.Box(
            low=-float('inf'),
            high=float('inf'),
            shape=[2],
        )


    def _convert_gymnasium_to_gym_box(self, box: gymnasium.spaces.Box) -> gym.spaces.Box:
        return gym.spaces.Box(
            low=box.low,
            high=box.high,
            dtype=box.dtype
        )


    def step(self, action):
        obs, reward, cost, terminated, truncated, info = self.env.step(action)
        info = {}
        info['cost'] = cost

        done = terminated or truncated

        if isinstance(obs, tuple):
            return obs[0], reward, done, info

        return obs, reward, done, info
    
    def reset(self):
        obs, info = self.env.reset()

        return obs
    
    @property
    def reward_space(self):
        return self._reward_space
    
    def render(self, mode="human"):
        self.env.render()


class GymnasiumWrapperVelocity(gym.Wrapper):
    def __init__(self, env):

        def seed(seed):
            ...

        super().__init__(env)

        self.seed = seed
        self.num_steps = 1000
        
        self.observation_space = self._convert_gymnasium_to_gym_box(self.observation_space)
        self.action_space = self._convert_gymnasium_to_gym_box(self.action_space)

        self._reward_space = gym.spaces.Box(
            low=-float('inf'),
            high=float('inf'),
            shape=[2],
        )


    def _convert_gymnasium_to_gym_box(self, box: gymnasium.spaces.Box) -> gym.spaces.Box:
        return gym.spaces.Box(
            low=box.low,
            high=box.high,
            dtype=box.dtype
        )


    def step(self, action):
        obs, reward, cost, terminated, truncated, info = self.env.step(action)
        info = {}

        done = terminated or truncated

        reward = np.array([reward, -cost], dtype=np.float32)

        if isinstance(obs, tuple):
            return obs[0], reward, done, info

        return obs, reward, done, info
    
    def reset(self):
        obs, _ = self.env.reset()
        return obs
    
    @property
    def reward_space(self):
        return self._reward_space
    
    def render(self, mode="human"):
        self.env.render()


class CompleteEnvInfo(gym.Wrapper):
    """Always set the complete set of information so that the env info has a
    fixed shape (no matter whether some event occurs or not), which is required
    by ALF.

    The current safety gym env only adds a key to env info when the corresponding
    event is triggered, see:
    https://github.com/openai/safety-gym/blob/f31042f2f9ee61b9034dd6a416955972911544f5/safety_gym/envs/engine.py#L1242
    """

    def __init__(self, env, env_name):
        super().__init__(env)
        # env info keys are retrieved from:
        # https://github.com/openai/safety-gym/blob/master/safety_gym/envs/engine.py
        self._env_info_keys = [
            'cost_exception',
            'goal_met',
            'cost'  # this is the summed overall cost
        ]
        if not self._is_level0_env(env_name):
            # for level 1 and 2 envs, there are constraints cost info
            self._env_info_keys += [
                'cost_vases_contact', 'cost_pillars', 'cost_buttons',
                'cost_gremlins', 'cost_vases_displace', 'cost_vases_velocity',
                'cost_hazards'
            ]
        self._default_env_info = self._generate_default_env_info()

    def _is_level0_env(self, env_name):
        return "0-v" in env_name

    def _generate_default_env_info(self):
        env_info = {}
        for key in self._env_info_keys:
            if key == "goal_met":
                env_info[key] = False
            else:
                env_info[key] = np.float32(0.)
        return env_info

    def step(self, action):
        """Take a step through the environment the returns the complete set of
        env info, regardless of whether the corresponding event is enabled or not.
        """
        env_info = copy.copy(self._default_env_info)
        obs, reward, done, info = self.env.step(action)
        env_info.update(info)
        return obs, reward, done, env_info


class VectorReward(gym.Wrapper):
    """This wrapper makes the env returns a reward vector of length 3. The three
    dimensions are:

    1. distance-improvement reward indicating the delta smaller distances of
       agent<->box and box<->goal for "push" tasks, or agent<->goal for
       "goal"/"button" tasks.
    2. negative binary cost where -1 means that at least one constraint has been
       violated at the current time step (constraints vary depending on env
       configurations).

    All rewards are the higher the better.
    """

    REWARD_DIMENSION = 2

    def __init__(self, env, sparse_reward):
        """
        Args:
            env: env being wrapped
            sparse_reward: if True, then the first reward dim will only be a
                binary value indicating a success.
        """
        super().__init__(env)
        self._reward_space = gym.spaces.Box(
            low=-float('inf'),
            high=float('inf'),
            shape=[self.REWARD_DIMENSION])
        self._sparse_reward = sparse_reward

    def step(self, action):
        """Take one step through the environment and obtains several rewards.

        Args:
            action (np.array):

        Returns:
            tuple:
            - obs (np.array): a flattened observation vector that contains
              all enabled sensors' data
            - rewards (np.array): a reward vector of length ``REWARD_DIMENSION``.
              See the class docstring for their meanings.
            - done (bool): whether the episode has ended
            - info (dict): a dict of additional env information
        """
        obs, reward, done, info = self.env.step(action)
        # Get the second and third reward from ``info``
        cost_reward = -info["cost"]
        success_reward = float(info["goal_met"])
        if self._sparse_reward:
            reward = success_reward
        return obs, np.array([reward, cost_reward],
                             dtype=np.float32), done, info

    @property
    def reward_space(self):
        return self._reward_space



@alf.configurable
class TestLogger(gym.Wrapper):
    def __init__(self, env, env_name, testing=False):
        super().__init__(env)
        self.testing = testing

        if not self.testing:
            self.writer = SummaryWriter(f"/home/user/siddiquieu1/HRL/SRCPO-v-SAC/tensorboard/seditor_{env_name}")
            self.step_count = 0
            self.acc_cost = 0
            self.acc_reward = 0
        

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        if not self.testing:
            self.acc_reward += reward[0]
            self.acc_cost += reward[1]
            self.step_count += 1

            if self.step_count % 1000 == 0:
                self.writer.add_scalar("Episode Cost", self.acc_cost, self.step_count / 1000)
                self.writer.add_scalar("Return", self.acc_reward, self.step_count / 1000)

                self.acc_cost = 0
                self.acc_reward = 0
        else:
            print("action", action)
            print("obs", obs)
            print("reward", reward)
            print("info", info)

        return obs, reward, done, info


@alf.configurable
def load(environment_name: str,
         env_id: int = None,
         discount: float = 1.0,
         max_episode_steps: int = None,
         unconstrained: bool = False,
         sparse_reward: bool = False,
         episodic: bool = False,
         gym_env_wrappers: List[Callable] = (),
         alf_env_wrappers: List[Callable] = ()):


    if TESTING:
        env = safety_gymnasium.make(environment_name, render_mode="human")
    else:
        env = safety_gymnasium.make(environment_name)

    # env = GymnasiumWrapper(env)
    env = GymnasiumWrapperVelocity(env)
    # env = CompleteEnvInfo(env, env_name=environment_name)
    # env = VectorReward(env, sparse_reward=False)
    env = TestLogger(env, env_name=environment_name, testing=TESTING)

    # print(env.action_space)
    # print(env.observation_space)
    # exit()

    if max_episode_steps is None:
        max_episode_steps = env.num_steps - 1
        max_episode_steps = min(env.num_steps - 1, max_episode_steps)

    return suite_gym.wrap_env(
        env,
        env_id=env_id,
        discount=discount,
        max_episode_steps=max_episode_steps,
        gym_env_wrappers=gym_env_wrappers,
        alf_env_wrappers=alf_env_wrappers)
