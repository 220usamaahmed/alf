try:
    import safety_gymnasium
except ImportError:
    safety_gymnasium = None

import gym
import gymnasium
from typing import Callable, List
import numpy as np
import copy
import os
import torch

from alf.environments.safety_critical.envs_critical import Glucose, BiGlucose, CSTR

import alf
from alf.environments import suite_gym
from torch.utils.tensorboard.writer import SummaryWriter

TESTING = 0
SEED = 0


class SafetyCriticalWrapper(gym.Wrapper):
    def __init__(self, env):

        def seed(seed):
            seed = SEED
            torch.manual_seed(seed)
            np.random.seed(seed)
            torch.cuda.manual_seed(seed)

        super().__init__(env)

        self.seed = seed
        self.num_steps = env.max_steps + 1

        self.observation_space = self._convert_gymnasium_to_gym_box(
            self.observation_space
        )
        self.action_space = self._convert_gymnasium_to_gym_box(self.action_space)

        self._reward_space = gym.spaces.Box(
            low=-float("inf"),
            high=float("inf"),
            shape=[2],
        )

    def _convert_gymnasium_to_gym_box(
        self, box: gymnasium.spaces.Box
    ) -> gym.spaces.Box:
        return gym.spaces.Box(low=box.low, high=box.high, dtype=box.dtype)

    def step(self, action):
        obs, reward, cost, terminated, truncated, info = self.env.step(action)
        info = {}

        # done = terminated or truncated
        done = truncated
        # if truncated:
        #     print("------------------------------- TRUNCATGED")

        reward = np.array([reward, -cost], dtype=np.float32)

        if isinstance(obs, tuple):
            return obs[0], reward, done, info

        return obs, reward, done, info

    def reset(self):
        self.curr_epi_lenght = 0
        obs, _ = self.env.reset()
        return obs

    @property
    def reward_space(self):
        return self._reward_space

    def render(self, mode="human"):
        self.env.render()


@alf.configurable
class TestLogger(gym.Wrapper):
    def __init__(self, env, env_name, seed, testing=False):
        super().__init__(env)
        self.testing = testing

        if not self.testing:
            ROOT = "/home/user/siddiquieu1/HRL/experiments"

            self.writer = SummaryWriter(
                os.path.join(
                    ROOT,
                    "runs",
                    env_name,
                    f"_{seed}",
                    "seditor",
                )
            )

            self.step_count = 0
            self.curr_episode_len = 0
            self.curr_episode_return = 0
            self.cumulative_cost = 0

    def step(self, action):

        obs, reward, done, info = self.env.step(action)

        if not self.testing:
            #

            self.step_count += 1
            self.curr_episode_len += 1
            self.curr_episode_return += reward[0]
            self.cumulative_cost += -reward[1]

            if self.step_count % 1000 == 0:
                self.writer.add_scalar(
                    "Cumulative_Cost", self.cumulative_cost, self.step_count
                )

            if done:
                self.writer.add_scalar(
                    "Episode_Len", self.curr_episode_len, self.step_count
                )
                self.writer.add_scalar(
                    "Episode_Return", self.curr_episode_return, self.step_count
                )

                print(
                    "Episode Length",
                    self.curr_episode_len,
                    "Steps",
                    self.step_count,
                    "Episode Return",
                    self.curr_episode_return,
                    "Cumulative Cost",
                    self.cumulative_cost,
                )

                self.curr_episode_len = 0
                self.curr_episode_return = 0

            #

        else:
            print("action", action)
            print("obs", obs)
            print("reward", reward)
            print("info", info)

        return obs, reward, done, info


@alf.configurable
def load(
    environment_name: str,
    env_id: int = None,
    discount: float = 1.0,
    max_episode_steps: int = None,
    unconstrained: bool = False,
    sparse_reward: bool = False,
    episodic: bool = False,
    gym_env_wrappers: List[Callable] = (),
    alf_env_wrappers: List[Callable] = (),
):

    # if env_id == "Glucose":
    #     env = Glucose()
    # elif env_id == "BiGlucose":
    #     env = BiGlucose()
    # elif env_id == "CSTR":
    #     env = CSTR()

    if environment_name == "Glucose":
        env = Glucose(altered_paras={"n": 0.2, "p2": 0.005, "p3": 5e-6})
    elif environment_name == "BiGlucose":
        env = BiGlucose(
            altered_paras={
                "D_G": 80,
                "V_G": 0.18,
                "k_12": 0.0343,
                "F_01": 0.0121,
                "EGP_0": 0.0148,
                "A_g": 0.8,
                "t_maxG": 40,
                "t_maxI": 55,
                "V_I": 0.12,
                "k_e": 0.138,
                "k_a1": 0.0031,
                "k_a2": 0.0752,
                "k_a3": 0.0472,
                "k_b1": 9.114e-06,
                "k_b2": 6.768e-06,
                "k_b3": 0.00189,
                "t_maxN": 32.46,
                "k_N": 0.62,
                "V_N": 16.06,
                "p": 0.016,
                "S_N": 19600.0,
                "M_g": 180.16,
                "BW": 68.5,
                "N_b": 48.13,
                "dt": 10,
            }
        )
    elif environment_name == "CSTR":
        env = CSTR(altered_paras={"alpha": 1.05, "beta": 1.1})

    # if TESTING:
    #     env = safety_gymnasium.make(environment_name, render_mode="human")
    # else:
    #     env = safety_gymnasium.make(environment_name)

    env = SafetyCriticalWrapper(env)
    env = TestLogger(env, env_name=environment_name, seed=SEED, testing=TESTING)

    if max_episode_steps is None:
        max_episode_steps = env.num_steps - 1
        max_episode_steps = min(env.num_steps - 1, max_episode_steps)

    return suite_gym.wrap_env(
        env,
        env_id=env_id,
        discount=discount,
        max_episode_steps=max_episode_steps,
        gym_env_wrappers=gym_env_wrappers,
        alf_env_wrappers=alf_env_wrappers,
    )
