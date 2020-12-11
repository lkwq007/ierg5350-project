import cv2
import numpy as np
import torch

from env_utils import make_envs

GYM_ENVS = ["TetrisA-v0","TetrisA-v1",'Pendulum-v0', 'MountainCarContinuous-v0', 'Ant-v2', 'HalfCheetah-v2', 'Hopper-v2', 'Humanoid-v2', 'HumanoidStandup-v2', 'InvertedDoublePendulum-v2', 'InvertedPendulum-v2', 'Reacher-v2', 'Swimmer-v2', 'Walker2d-v2']
CONTROL_SUITE_ENVS = ['cartpole-balance', 'cartpole-swingup', 'reacher-easy', 'finger-spin', 'cheetah-run', 'ball_in_cup-catch', 'walker-walk','reacher-hard', 'walker-run', 'humanoid-stand', 'humanoid-walk', 'fish-swim', 'acrobot-swingup']
CONTROL_SUITE_ACTION_REPEATS = {'cartpole': 8, 'reacher': 4, 'finger': 2, 'cheetah': 4, 'ball_in_cup': 6, 'walker': 2, 'humanoid': 2, 'fish': 2, 'acrobot':4}


# Preprocesses an observation inplace (from float32 Tensor [0, 255] to [-0.5, 0.5])
def preprocess_observation_(observation, bit_depth):
  observation.div_(2 ** (8 - bit_depth)).floor_().div_(2 ** bit_depth).sub_(0.5)  # Quantise to given bit depth and centre
  observation.add_(torch.rand_like(observation).div_(2 ** bit_depth))  # Dequantise (to approx. match likelihood of PDF of continuous images vs. PMF of discrete images)


# Postprocess an observation for storage (from float32 numpy array [-0.5, 0.5] to uint8 numpy array [0, 255])
def postprocess_observation(observation, bit_depth):
  return np.clip(np.floor((observation + 0.5) * 2 ** bit_depth) * 2 ** (8 - bit_depth), 0, 2 ** 8 - 1).astype(np.uint8)


def _images_to_observation(images, bit_depth):
  images = torch.tensor(cv2.resize(images, (96, 96), interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1), dtype=torch.float32)  # Resize and put channel first
  preprocess_observation_(images, bit_depth)  # Quantise, centre and dequantise inplace
  return images.unsqueeze(dim=0)  # Add batch dimension


class ControlSuiteEnv():
  def __init__(self, env, symbolic, seed, max_episode_length, action_repeat, bit_depth):
    from dm_control import suite
    from dm_control.suite.wrappers import pixels
    domain, task = env.split('-')
    self.symbolic = symbolic
    self._env = suite.load(domain_name=domain, task_name=task, task_kwargs={'random': seed})
    if not symbolic:
      self._env = pixels.Wrapper(self._env)
    self.max_episode_length = max_episode_length
    self.action_repeat = action_repeat
    if action_repeat != CONTROL_SUITE_ACTION_REPEATS[domain]:
      print('Using action repeat %d; recommended action repeat for domain is %d' % (action_repeat, CONTROL_SUITE_ACTION_REPEATS[domain]))
    self.bit_depth = bit_depth

  def reset(self):
    self.t = 0  # Reset internal timer
    state = self._env.reset()
    if self.symbolic:
      return torch.tensor(np.concatenate([np.asarray([obs]) if isinstance(obs, float) else obs for obs in state.observation.values()], axis=0), dtype=torch.float32).unsqueeze(dim=0)
    else:
      return _images_to_observation(self._env.physics.render(camera_id=0), self.bit_depth)

  def step(self, action):
    action = action.detach().numpy()
    reward = 0
    for k in range(self.action_repeat):
      state = self._env.step(action)
      reward += state.reward
      self.t += 1  # Increment internal timer
      done = state.last() or self.t == self.max_episode_length
      if done:
        break
    if self.symbolic:
      observation = torch.tensor(np.concatenate([np.asarray([obs]) if isinstance(obs, float) else obs for obs in state.observation.values()], axis=0), dtype=torch.float32).unsqueeze(dim=0)
    else:
      observation = _images_to_observation(self._env.physics.render(camera_id=0), self.bit_depth)
    return observation, reward, done

  def render(self):
    cv2.imshow('screen', self._env.physics.render(camera_id=0)[:, :, ::-1])
    cv2.waitKey(1)

  def close(self):
    cv2.destroyAllWindows()
    self._env.close()

  @property
  def observation_size(self):
    return sum([(1 if len(obs.shape) == 0 else obs.shape[0]) for obs in self._env.observation_spec().values()]) if self.symbolic else (3, 96, 96)

  @property
  def action_size(self):
    return self._env.action_spec().shape[0]

  # Sample an action randomly from a uniform distribution over all valid actions
  def sample_random_action(self):
    spec = self._env.action_spec()
    return torch.from_numpy(np.random.uniform(spec.minimum, spec.maximum, spec.shape))

import gym
from nes_py.wrappers import JoypadSpace
import gym_tetris
from gym_tetris.actions import MOVEMENT,SIMPLE_MOVEMENT
from gym import spaces
import cv2

def crop_image(image):
    '''
    Args: image as an numpy array
    returns: a crop image as a standardised numpy array
    '''
    image = image[47:209,95:176,:]
    image = np.mean(image, axis=2)
    image[image > 0] = 1
    image = cv2.resize(image, (10,20))
    image = image.astype(np.float32)
    return image
  
class SymbolTetris(gym.Wrapper):
  def __init__(self, env):
    gym.Wrapper.__init__(self, env)
    shp = env.observation_space.shape
    # set 
    self.observation_space = spaces.Box(
      low=0.0,
      high=1.0,
      shape=(200,),
      dtype=np.float32
    )

  def reset(self):
    obs = self.env.reset()
    return self._get_board(obs)
  
  def _get_board(self,obs):
    return crop_image(obs).reshape(-1)
  
  def _get_board_ram(self):
    board=self.env.ram[0x0400:0x04C8].copy()
    board[board == 239] = 0
    board[board>1]=1
    board=board.reshape(20,10).astype(np.float32)
    return board

  def step(self, action):            
    obs, reward, done, info = self.env.step(action)
    return self._get_board(obs), reward, done, info

def get_bumpiness_height_hole(board):
  board = np.array(board)
  mask = board > 0
  invert_heights = np.where(mask.any(axis=0), np.argmax(mask, axis=0), 20)
  heights = 20 - invert_heights
  total_height = np.sum(heights)
  currs = heights[:-1]
  nexts = heights[1:]
  diffs = np.abs(currs - nexts)
  total_bumpiness = np.sum(diffs)
  total_cell=np.sum(board)
  return total_bumpiness, total_height, total_height-total_cell

from torch.nn import functional as F
class GymEnv():
  def __init__(self, env, symbolic, seed, max_episode_length, action_repeat, bit_depth, args):
    self.symbolic = symbolic
    # self._env = gym.make(env)
    self._env = gym_tetris.make(env,skip_level=True)
    self._env.seed(seed)
    self._env = JoypadSpace(self._env, SIMPLE_MOVEMENT)
    if symbolic:
      self._env=SymbolTetris(self._env)
    self.max_episode_length = max_episode_length
    self.action_repeat = 1
    self.bit_depth = bit_depth
    self.typeb = "1" in env
    self.acc = 0.01 if self.typeb else 3
    self.living = 0.001 if self.typeb else 0.3
    self.die=-10
    self.score=0.0
    self.add_reward=args.add_reward
    if not args.add_reward:
      self.acc=0
      self.living=0
      self.die=0

  def reset(self):
    self.t = 0  # Reset internal timer
    state = self._env.reset()
    self.score=0.0
    # self._env.ram[0x0064]=29
    for i in range(85):
      state,r,d,i=self._env.step(0)
    if self.symbolic:
      return torch.tensor(state, dtype=torch.float32).unsqueeze(dim=0)
    else:
      return _images_to_observation(state[46:-30,92:-12,:], self.bit_depth)

  def _get_board_ram(self):
    board=self._env.ram[0x0400:0x04C8].copy()
    board[board == 239] = 0
    board[board>1]=1
    board=board.reshape(20,10).astype(np.float32)
    return board

  def step(self, action):
    # action = action.detach().numpy()
    action = action.argmax().item()
    reward = 0
    state, done = None, None
    last_board=self._get_board_ram()
    for k in range(3):
      state, reward_k, done, _ = self._env.step(action if k==0 else 0)
      reward += reward_k
      self.t += 1  # Increment internal timer
      done = done or self.t == self.max_episode_length
      if done:
        break
    flag=False
    if reward>0:
      board=last_board
    else:
      board=self._get_board_ram()
    while self._env.ram[0x0065]>0 and self._env.ram[0x0068]>=2 and not done:
      flag=True
      state,r,d,info=self._env.step(0)
      reward+=r
      done=d or done
    if (flag or reward>0) and self.add_reward:
      bumpiness, heights, holes=get_bumpiness_height_hole(board)
      score = 0.76*reward - 0.51*heights - 0.36*holes - 0.18*bumpiness
      reward = score-self.score
      self.score=score
      # reward+=self.acc
      # if info['board_height']>10:
      #   reward-=self.acc
    reward+=self.living if action==0 else self.living/2
    if done:
      reward+=self.die
    if self.symbolic:
      observation = torch.tensor(state, dtype=torch.float32).unsqueeze(dim=0)
    else:
      observation = _images_to_observation(state[46:-30,92:-12,:], self.bit_depth)
    return observation, reward, done

  def render(self):
    self._env.render()

  def close(self):
    self._env.close()

  @property
  def observation_size(self):
    return self._env.observation_space.shape[0] if self.symbolic else (3, 96, 96)

  @property
  def action_size(self):
    return self._env.action_space.n

  # Sample an action randomly from a uniform distribution over all valid actions
  def sample_random_action(self):
    indices = torch.tensor(self._env.action_space.sample())
    return F.one_hot(indices, self.action_size).float()
    # return torch.tensor(self._env.action_space.sample())


def Env(env, symbolic, seed, max_episode_length, action_repeat, bit_depth, args):
  if env in GYM_ENVS:
    return GymEnv(env, symbolic, seed, max_episode_length, action_repeat, bit_depth, args)
  elif env in CONTROL_SUITE_ENVS:
    return ControlSuiteEnv(env, symbolic, seed, max_episode_length, action_repeat, bit_depth)


# Wrapper for batching environments together
class EnvBatcher():
  def __init__(self, env_class, env_args, env_kwargs, n):
    self.n = n
    self.envs = [env_class(*env_args, **env_kwargs) for _ in range(n)]
    self.dones = [True] * n
    self.shp=(1,200)

  # Resets every environment and returns observation
  def reset(self):
    observations = [env.reset() for env in self.envs]
    self.dones = [False] * self.n
    return torch.cat(observations)

 # Steps/resets every environment and returns (observation, reward, done)
  def step(self, actions):
    done_mask = torch.nonzero(torch.tensor(self.dones))[:, 0]  # Done mask to blank out observations and zero rewards for previously terminated environments
    observations, rewards, dones = zip(
            *[env.step(action) if not d else (torch.zeros(*self.shp), 0,True) for env, action, d in zip(self.envs, actions, self.dones)])
    # observations, rewards, dones = zip(*[env.step(action) for env, action in zip(self.envs, actions)])
    dones = [d or prev_d for d, prev_d in zip(dones, self.dones)]  # Env should remain terminated if previously terminated
    self.dones = dones
    observations, rewards, dones = torch.cat(observations), torch.tensor(rewards, dtype=torch.float32), torch.tensor(dones, dtype=torch.uint8)
    observations[done_mask] = 0
    rewards[done_mask] = 0
    return observations, rewards, dones

  def close(self):
    [env.close() for env in self.envs]
