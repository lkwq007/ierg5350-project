import numpy as np
import gym
from gym import spaces
import random
from . import tetris_engine as game

SCREEN_WIDTH, SCREEN_HEIGHT = 50, 100


class TetrisEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self):
        # open up a game state to communicate with emulator
        self.game_state = game.GameState()
        self._action_set = self.game_state.getActionSet()
        self.action_space = spaces.Discrete(len(self._action_set))
        self.observation_space = spaces.Box(low=0, high=255, shape=(SCREEN_HEIGHT, SCREEN_WIDTH, 3))
        self.viewer = None
      
    def seed(self,seed):
      random.seed(seed)
      return [seed]


    def step(self, a):
        self._action_set = np.zeros([len(self._action_set)])
        self._action_set[a] = 1
        reward = 0.0
        state, reward, terminal = self.game_state.frame_step(self._action_set)
        return state.transpose(1,0,2), reward, terminal, {}

    def _get_image(self):
        return self.game_state.getImage()

    @property
    def _n_actions(self):
        return len(self._action_set)

    # return: (states, observations)
    def reset(self):
        # random.seed(self._seed)
        do_nothing = np.zeros(len(self._action_set))
        do_nothing[0] = 1
        self.observation_space = spaces.Box(low=0, high=255, shape=(SCREEN_HEIGHT, SCREEN_WIDTH, 3))
        state, _, _= self.game_state.frame_step(do_nothing)
        return state.transpose(1,0,2)

    def render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        img = self._get_image()
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
