import copy
import math
import os
import pickle as pkl
import sys
import time
import gym
from gym import spaces
import numpy as np

# import dmc2gym
import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from logger import Logger
from replay_buffer import ReplayBuffer
from video import VideoRecorder

torch.backends.cudnn.benchmark = True

class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,)+env.observation_space.shape, dtype=np.uint8)
        self._skip       = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            if i==0:
                act=action
            else:
                act=0
            obs, reward, done, info = self.env.step(act)
            if i == self._skip - 2: self._obs_buffer[0] = obs
            if i == self._skip - 1: self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

class WrapPyTorch(gym.ObservationWrapper):
    def __init__(self, env=None, max_episode_steps=10000):
        super(WrapPyTorch, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = spaces.Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [obs_shape[2], obs_shape[0], obs_shape[1]],
            dtype=self.observation_space.dtype)
        self._max_episode_steps = max_episode_steps
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
        return np.array(frame).transpose(2, 0, 1)

from nes_py.wrappers import JoypadSpace
import gym_tetris
from gym_tetris.actions import MOVEMENT, SIMPLE_MOVEMENT


def make_env(cfg):
    """Helper function to create dm_control environment"""
    if cfg.env == 'ball_in_cup_catch':
        domain_name = 'ball_in_cup'
        task_name = 'catch'
    elif cfg.env == 'point_mass_easy':
        domain_name = 'point_mass'
        task_name = 'easy'
    else:
        domain_name = cfg.env.split('_')[0]
        task_name = '_'.join(cfg.env.split('_')[1:])

    # per dreamer: https://github.com/danijar/dreamer/blob/02f0210f5991c7710826ca7881f19c64a012290c/wrappers.py#L26
    camera_id = 2 if domain_name == 'quadruped' else 0

#     env = dmc2gym.make(domain_name=domain_name,
#                        task_name=task_name,
#                        seed=cfg.seed,
#                        visualize_reward=False,
#                        from_pixels=True,
#                        height=cfg.image_size,
#                        width=cfg.image_size,
#                        frame_skip=cfg.action_repeat,
#                        camera_id=camera_id)
    # env = gym.make("CarRacing-v0")
    env_ = gym_tetris.make('TetrisA-v0')
    env = JoypadSpace(env_, SIMPLE_MOVEMENT)
    # env = MaxAndSkipEnv(env)
    # env._max_episode_steps = env_._max_episode_steps
    max_episode_steps = 10000
    env = WrapPyTorch(env, max_episode_steps)
    env.seed(cfg.seed)
    # print(env.ram)
    obs = env.reset()
    print(obs.shape)
    # env.seed(cfg.seed)

    env = utils.FrameStack(env, k=cfg.frame_stack)
    print("Init env done")
    # assert env.action_space.low.min() >= -1
    # assert env.action_space.high.max() <= 1

    return env


class Workspace(object):
    def __init__(self, cfg):
        self.work_dir = os.getcwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg

        self.logger = Logger(self.work_dir,
                             save_tb=cfg.log_save_tb,
                             log_frequency=cfg.log_frequency_step,
                             agent=cfg.agent.name,
                             action_repeat=cfg.action_repeat)

        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.env = make_env(cfg)

        cfg.agent.params.obs_shape = self.env.observation_space.shape
        print(self.env.action_space.shape)
        cfg.agent.params.action_shape = (self.env.action_space.n,)
        cfg.agent.params.action_range = [
            0,
            12
        ]
        self.agent = hydra.utils.instantiate(cfg.agent)

        self.replay_buffer = ReplayBuffer(self.env.observation_space.shape,
                                          self.env.action_space.shape,
                                          cfg.replay_buffer_capacity,
                                          self.cfg.image_pad, self.device)

        self.video_recorder = VideoRecorder(
            self.work_dir if cfg.save_video else None)
        self.step = 0

    def evaluate(self):
        average_episode_reward = 0
        for episode in range(self.cfg.num_eval_episodes):
            obs = self.env.reset()
            self.video_recorder.init(enabled=(episode == 0))
            done = False
            episode_reward = 0
            episode_step = 0
            print(episode)
            while not done:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=False)
                # print(acion.shape)
                action=action[0]
                obs, reward, done, info = self.env.step(action)
                self.video_recorder.record(obs)
                episode_reward += reward
                episode_step += 1
                if episode_step>10000:
                    break

            average_episode_reward += episode_reward
            self.video_recorder.save(f'{self.step}.mp4')
        average_episode_reward /= self.cfg.num_eval_episodes
        self.logger.log('eval/episode_reward', average_episode_reward,
                        self.step)
        self.logger.dump(self.step)

    def run(self):
        episode, episode_reward, episode_step, done = 0, 0, 1, True
        start_time = time.time()
        while self.step < self.cfg.num_train_steps:
            if done:
                if self.step > 0:
                    self.logger.log('train/duration',
                                    time.time() - start_time, self.step)
                    start_time = time.time()
                    self.logger.dump(
                        self.step, save=(self.step > self.cfg.num_seed_steps))

                # evaluate agent periodically
                if self.step % self.cfg.eval_frequency == 0:
                    self.logger.log('eval/episode', episode, self.step)
                    self.evaluate()

                self.logger.log('train/episode_reward', episode_reward,
                                self.step)

                obs = self.env.reset()
                done = False
                episode_reward = 0
                episode_step = 0
                episode += 1

                self.logger.log('train/episode', episode, self.step)

            # sample action for data collection
            if self.step >= self.cfg.num_seed_steps:
                action = self.env.action_space.sample()
                # action=np.array([action])
            else:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=True)
                    action=action[()]

            # run training update
            if self.step >= self.cfg.num_seed_steps:
                for _ in range(self.cfg.num_train_iters):
                    self.agent.update(self.replay_buffer, self.logger,
                                      self.step)

            next_obs, reward, done, info = self.env.step(action)

            # allow infinite bootstrap
            done = float(done)
            done_no_max = 0 if episode_step + 1 == self.env._max_episode_steps else done
            episode_reward += reward

            self.replay_buffer.add(obs, action, reward, next_obs, done,
                                   done_no_max)

            obs = next_obs
            episode_step += 1
            self.step += 1


@hydra.main(config_path='config.yaml', strict=True)
def main(cfg):
    from train import Workspace as W
    workspace = W(cfg)
    workspace.run()


if __name__ == '__main__':
    main()
