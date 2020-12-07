"""

Taken files from competitive_rl. I think these source codes belong to openai.baseline

This file defines a helper function to build our environment.

Usages:
    make_envs(
        env_id="cPong-v0",      # Environment name, must in [
                                    "cPongTournament-v0",
                                    "cPongDouble-v0",
                                    "cPong-v0",
                                    "CartPole-v0"
                                    ].
        seed=0,                 # Random seed
        log_dir="data",         # Which directory to store data and checkpoints
        num_envs=5,             # How many concurrent environments to run
        asynchronous=True,      # Whether to use asynchronous envrionments.
                                # This can extremely accelerate the system
        resized_dim=42          # Resized the observation to a 42x42 image
    )

Notes:
    1. If you wish to use asynchronous environments, you should run it in python
        scripts under "if __name__ == '__main__'" line.
    2. CartPole-v0 environment can be used for testing algorithms.
"""
import os
import shutil
import warnings

import gym
from gym import spaces

from nes_py.wrappers import JoypadSpace
from gym_tetris.actions import MOVEMENT, SIMPLE_MOVEMENT

from .dummy_vec_env import DummyVecEnv
from .subproc_vec_env import SubprocVecEnv

__all__ = ["make_envs"]

msg = """
Multiprocessing vectorized environments need to be created under 
"if __name__ == '__main__'" line due to the limitation of multiprocessing 
module. 

If you are testing codes within interactive interface like jupyter notebook, please set the num_envs to 1, 
i.e. make_envs(num_envs=1) to avoid such error. We return envs = None now.
"""


class WrapPyTorch(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(WrapPyTorch, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = spaces.Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [obs_shape[2], obs_shape[0], obs_shape[1]],
            dtype=self.observation_space.dtype)
        if isinstance(self.env.unwrapped.observation_space, spaces.Tuple):
            self.observation_space = spaces.Tuple(
                [self.observation_space, self.observation_space])

    def observation(self, observation):
        if isinstance(observation, tuple):
            return tuple(self.parse_single_frame(f) for f in observation)
        elif isinstance(observation, dict):
            return {k: self.parse_single_frame(f) for k, f in observation.items()}
        else:
            return self.parse_single_frame(observation)

    def parse_single_frame(self, frame):
        assert frame.ndim == 3
        return frame.transpose(2, 0, 1)


def make_envs(env_id="cPong-v0", seed=0, log_dir="data", num_envs=3, asynchronous=False, resized_dim=42, frame_stack=4,
              action_repeat=None, simple_movement=False):
    """
    Create CUHKPong-v0, CUHKPongDouble-v0 or CartPole-v0 environments. If
    num_envs > 1, put them into different processes.

    :param env_id: The name of environment you want to create
    :param seed: The random seed
    :param log_dir: The path to store the learning stats
    :param num_envs: How many environments you want to run concurrently (Too
        large number will block your program.)
    :param asynchronous: whether to use multiprocessing
    :param resized_dim: resize the observation to image with shape (1,
        resized_dim, resized_dim)
    :return: A vectorized environment
    """
    asynchronous = asynchronous and num_envs > 1

    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    def env_wrapper(env_id, seed=0, rank=0, action_repeat=1):
        def _make():
            env = gym.make(env_id)
            env.seed(seed+rank)
            move = SIMPLE_MOVEMENT if simple_movement else MOVEMENT
            env = JoypadSpace(env, move)
            env = WrapPyTorch(env)
            return env
        return _make

    envs = [env_wrapper(env_id, seed=seed, rank=i,
                        action_repeat=action_repeat) for i in range(num_envs)]

    if asynchronous:
        envs = SubprocVecEnv(envs)
    else:
        envs = DummyVecEnv(envs)
    return envs


if __name__ == '__main__':
    # Testing
    tournament_envs = make_envs("cPongTournament-v0", num_envs=3,
                                log_dir="tmp", asynchronous=True)
    tournament_envs.reset()
    tournament_envs.step([0, 1, 2])

    double_envs = make_envs("cPongDouble-v0", num_envs=3,
                            log_dir="tmp", asynchronous=True)
    double_envs.reset()
    double_obs_a, double_rew_a, double_done_a, double_info_a = double_envs.step(
        [[0, 0], [1, 0], [2, 1]])

    double_envs = make_envs("cPongDouble-v0", num_envs=3,
                            log_dir="tmp", asynchronous=False)
    double_envs.reset()
    double_obs, double_rew, double_done, double_info = double_envs.step(
        [[0, 0], [1, 0], [2, 1]])

    envs = make_envs("cPong-v0", num_envs=3, log_dir="tmp",
                     asynchronous=False)
    envs.reset()
    obs, rew, done, info = envs.step([0, 1, 2])

    # Test consistency between cPongTournament and cPong
    envs = make_envs("cPong-v0", num_envs=3, log_dir="tmp",
                     asynchronous=False)

    tournament_envs = make_envs("cPongTournament-v0", num_envs=3,
                                log_dir="tmp", asynchronous=False)
    assert envs.reset().shape == tournament_envs.reset().shape
    o1, r1, d1, i1 = envs.step([0, 1, 0])
    o2, r2, d2, i2 = tournament_envs.step([0, 1, 0])
    assert o1.shape == o2.shape
    assert r1.shape == r2.shape, (r1.shape, r2.shape)
    assert d1.shape == d2.shape, (d1.shape, d2.shape)

    envs = make_envs("cPong-v0", num_envs=1, log_dir="tmp",
                     asynchronous=False)
    tournament_envs = make_envs("cPongTournament-v0", num_envs=1,
                                log_dir="tmp", asynchronous=False)
    assert envs.reset().shape == tournament_envs.reset().shape
    o1, r1, d1, i1 = envs.step([0])
    o2, r2, d2, i2 = tournament_envs.step([0])
    assert o1.shape == o2.shape
    assert r1.shape == r2.shape, (r1.shape, r2.shape)
    assert d1.shape == d2.shape, (d1.shape, d2.shape)

    shutil.rmtree("./tmp", ignore_errors=True)
