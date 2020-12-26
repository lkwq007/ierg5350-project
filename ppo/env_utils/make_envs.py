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

from collections import deque

import cv2
import numpy as np
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
import cv2

class FrameStack(gym.Wrapper):
    def __init__(self, env, n_frames):
        """Stack n_frames last frames.

        Returns lazy array, which is much more memory efficient.

        See Also
        --------
        stable_baselines.common.atari_wrappers.LazyFrames

        :param env: (Gym Environment) the environment
        :param n_frames: (int) the number of frames to stack
        """
        gym.Wrapper.__init__(self, env)
        self.n_frames = n_frames
        self.frames = deque([], maxlen=n_frames)
        shp = env.observation_space.shape
        # set 
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(96, 96, shp[2] * n_frames),
            dtype=env.observation_space.dtype
        )

    def reset(self):
        obs = self.env.reset()
#         self.env.ram[0x0064]=29
        # skip frames
        for i in range(85):
            obs,r,d,i = self.env.step(0)
        _obs=self._preprocess(obs)
        for _ in range(self.n_frames):
            self.frames.append(_obs)
        return self._get_ob()

    def step(self, action):
        reward=0
        done=False
        for i in range(3):
            obs, r, d, info = self.env.step(action if i==0 else 0)
            reward+=r
            done=d or done
            if done:
                break
        self.frames.append(self._preprocess(obs))
        flag=False
        while self.env.ram[0x0065]>0 and self.env.ram[0x0068]>=2 and not done:
            flag=True
            o,r,d,info=self.env.step(0)
            reward+=r
            done=d or done
#         if flag and not done:
#             reward+=5
        if done:
            reward-=800
        else:
            reward+=5
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.n_frames
        return np.concatenate(self.frames, axis=2)
    
    def _preprocess(self,obs):
        return cv2.resize(obs[46:-30,92:-12,:], (96, 96), interpolation=cv2.INTER_CUBIC)

class FrameStack2(gym.Wrapper):
    def __init__(self, env, n_frames):
        """Stack n_frames last frames.

        Returns lazy array, which is much more memory efficient.

        See Also
        --------
        stable_baselines.common.atari_wrappers.LazyFrames

        :param env: (Gym Environment) the environment
        :param n_frames: (int) the number of frames to stack
        """
        gym.Wrapper.__init__(self, env)
        self.n_frames = n_frames
        self.frames = deque([], maxlen=n_frames)
        shp = env.observation_space.shape
        # set 
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(shp[0], shp[1], shp[2] * n_frames),
            dtype=env.observation_space.dtype
        )

    def reset(self):
        obs = self.env.reset()
        for _ in range(self.n_frames):
            self.frames.append(obs)
        return self._get_ob()

    def step(self, action):            
        obs, reward, done, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.n_frames
        return np.concatenate(self.frames, axis=2)


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

SIMPLE_MOVEMENT2 = [
    ['NOOP'],
    ['A'],
    ['B'],
    ['right'],
    ['left'],
]
def make_envs(env_id="cPong-v0", seed=0, log_dir="data", num_envs=3, asynchronous=False, resized_dim=42, frame_stack=4,
              action_repeat=None, simple_movement=True):
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
            move = SIMPLE_MOVEMENT2 if simple_movement else MOVEMENT
            env = JoypadSpace(env, move)
            env = FrameStack(env, frame_stack)
            env = WrapPyTorch(env)
            return env
        return _make
    def env_wrapper2(env_id, seed=0, rank=0, action_repeat=1):
        def _make():
            import gym_tetris_simple
            env = gym.make(env_id)
            env.seed(seed+rank)
            env = FrameStack2(env, frame_stack)
            env = WrapPyTorch(env)
            return env
        return _make
    if env_id=="Tetris-v0":
        print("Using simple Tetris")
        envs = [env_wrapper2(env_id, seed=seed, rank=i,
                        action_repeat=action_repeat) for i in range(num_envs)]
    else:
        envs = [env_wrapper(env_id, seed=seed, rank=i,
                        action_repeat=action_repeat) for i in range(num_envs)]

    if asynchronous:
        envs = SubprocVecEnv(envs)
    else:
        envs = DummyVecEnv(envs)
    return envs


if __name__ == '__main__':
    # Testing
    pass