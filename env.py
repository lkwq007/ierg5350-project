import cv2
import numpy as np
import torch
from torch.nn import functional as F
from env_utils import make_envs

NES_ENVS = ['TetrisA-v0', 'TetrisA-v1', 'TetrisB-v0', 'TetrisB-v1']

GYM_ENVS = [
    'Pendulum-v0', 'MountainCarContinuous-v0', 'Ant-v2', 'HalfCheetah-v2',
    'Hopper-v2', 'Humanoid-v2', 'HumanoidStandup-v2',
    'InvertedDoublePendulum-v2', 'InvertedPendulum-v2', 'Reacher-v2',
    'Swimmer-v2', 'Walker2d-v2'
]
CONTROL_SUITE_ENVS = [
    'cartpole-balance', 'cartpole-swingup', 'reacher-easy', 'finger-spin',
    'cheetah-run', 'ball_in_cup-catch', 'walker-walk', 'reacher-hard',
    'walker-run', 'humanoid-stand', 'humanoid-walk', 'fish-swim',
    'acrobot-swingup'
]
CONTROL_SUITE_ACTION_REPEATS = {
    'cartpole': 8,
    'reacher': 4,
    'finger': 2,
    'cheetah': 4,
    'ball_in_cup': 6,
    'walker': 2,
    'humanoid': 2,
    'fish': 2,
    'acrobot': 4
}

# that would be quite slow
# Preprocesses an observation inplace (from float32 Tensor [0, 255] to [-0.5, 0.5])
def preprocess_observation_(observation, bit_depth):
    observation.div_(2**(8 - bit_depth)).floor_().div_(2**bit_depth).sub_(
        0.5)  # Quantise to given bit depth and centre
    observation.add_(
        torch.rand_like(observation).div_(2**bit_depth)
    )  # Dequantise (to approx. match likelihood of PDF of continuous images vs. PMF of discrete images)


# Postprocess an observation for storage (from float32 numpy array [-0.5, 0.5] to uint8 numpy array [0, 255])
def postprocess_observation(observation, bit_depth):
    return np.clip(
        np.floor((observation + 0.5) * 2**bit_depth) * 2**(8 - bit_depth), 0,
        2**8 - 1).astype(np.uint8)


def _images_to_observation(images, bit_depth, img_size=(3, 128, 128)):
    images = torch.FloatTensor(
        cv2.resize(images, (img_size[2], img_size[1]),
                   interpolation=cv2.INTER_LINEAR).transpose(
                       2, 0, 1))  # Resize and put channel first
    preprocess_observation_(
        images, bit_depth)  # Quantise, centre and dequantise inplace
    return images.unsqueeze(dim=0)  # Add batch dimension


class ControlSuiteEnv():
    def __init__(self, env, seed, max_episode_length, action_repeat,
                 bit_depth):
        from dm_control import suite
        from dm_control.suite.wrappers import pixels
        domain, task = env.split('-')
        self._env = suite.load(domain_name=domain,
                               task_name=task,
                               task_kwargs={'random': seed})
        self._env = pixels.Wrapper(self._env)
        self.max_episode_length = max_episode_length
        self.action_repeat = action_repeat
        if action_repeat != CONTROL_SUITE_ACTION_REPEATS[domain]:
            print(
                'Using action repeat %d; recommended action repeat for domain is %d'
                % (action_repeat, CONTROL_SUITE_ACTION_REPEATS[domain]))
        self.bit_depth = bit_depth

    def reset(self):
        self.t = 0  # Reset internal timer
        state = self._env.reset()
        return _images_to_observation(self._env.physics.render(camera_id=0),
                                      self.bit_depth)

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
        observation = _images_to_observation(
            self._env.physics.render(camera_id=0), self.bit_depth)
        return observation, reward, done

    def render(self):
        cv2.imshow('screen', self._env.physics.render(camera_id=0)[:, :, ::-1])
        cv2.waitKey(1)

    def close(self):
        cv2.destroyAllWindows()
        self._env.close()

    @property
    def observation_size(self):
        return (3, 128, 128)

    @property
    def action_size(self):
        return self._env.action_spec().shape[0]

    # Sample an action randomly from a uniform distribution over all valid actions
    def sample_random_action(self):
        spec = self._env.action_spec()
        return torch.from_numpy(
            np.random.uniform(spec.minimum, spec.maximum, spec.shape))


class NesEnv():
    def __init__(self, env, seed, max_episode_length, action_repeat,
                 bit_depth):
        from nes_py.wrappers import JoypadSpace
        import gym_tetris
        from gym_tetris.actions import SIMPLE_MOVEMENT

        self._env = gym_tetris.make(env)
        self._env.seed(seed)
        self._env = JoypadSpace(self._env, SIMPLE_MOVEMENT)
        self.max_episode_length = max_episode_length
        self.action_repeat = action_repeat
        self.bit_depth = bit_depth

    def reset(self):
        self.t = 0  # Reset internal timer
        state = self._env.reset()
        # hack the memory of the nes env, setting level to 29
        self._env.ram[0x0064]=29
        # skip some frames
        for i in range(85):
            state,r,d,i=self._env.step(0)
        observation = _images_to_observation(state, self.bit_depth,
                                             self.observation_size)  # NxCxHxW
        return observation

    def step(self, action):
        action = action.argmax().item()  # convert onehot action to int
        reward = 0
        state, done = None, None
        for k in range(self.action_repeat):
            state, reward_k, done, _ = self._env.step(action)
            reward += reward_k
            self.t += 1  # Increment internal timer
            done = done or self.t == self.max_episode_length
            if done:
                break
        observation = _images_to_observation(state, self.bit_depth,
                                             self.observation_size)
        return observation, reward, done

    def render(self):
        self._env.render()

    def close(self):
        self._env.close()

    @property
    def observation_size(self):
        # self._env.observation_space.shape: H x W x C (240x256x3)
        return (3, 128, 128)  # C x H x W
        # return (3, 120, 128) # C x H x W # TODO: Lixin

    @property
    def action_size(self):
        return self._env.action_space.n

    def sample_random_action(self):
        indices = torch.tensor(self._env.action_space.sample())
        return F.one_hot(indices, self.action_size).float()


class GymEnv():
    def __init__(self, env, seed, max_episode_length, action_repeat,
                 bit_depth):
        import gym
        self._env = gym.make(env)
        self._env.seed(seed)
        self.max_episode_length = max_episode_length
        self.action_repeat = action_repeat
        self.bit_depth = bit_depth

    def reset(self):
        self.t = 0  # Reset internal timer
        state = self._env.reset()
        return _images_to_observation(self._env.render(mode='rgb_array'),
                                      self.bit_depth)

    def step(self, action):
        action = action.detach().numpy()
        reward = 0
        for k in range(self.action_repeat):
            state, reward_k, done, _ = self._env.step(action)
            reward += reward_k
            self.t += 1  # Increment internal timer
            done = done or self.t == self.max_episode_length
            if done:
                break
        observation = _images_to_observation(
            self._env.render(mode='rgb_array'), self.bit_depth)
        return observation, reward, done

    def render(self):
        self._env.render()

    def close(self):
        self._env.close()

    @property
    def observation_size(self):
        return (3, 128, 128)

    @property
    def action_size(self):
        return self._env.action_space.shape[0]

    # Sample an action randomly from a uniform distribution over all valid actions
    def sample_random_action(self):
        return torch.from_numpy(self._env.action_space.sample())


def Env(env, seed, max_episode_length, action_repeat, bit_depth):
    if env in GYM_ENVS:
        return GymEnv(env, seed, max_episode_length, action_repeat, bit_depth)
    elif env in CONTROL_SUITE_ENVS:
        return ControlSuiteEnv(env, seed, max_episode_length, action_repeat,
                               bit_depth)
    elif env in NES_ENVS:
        return NesEnv(env, seed, max_episode_length, action_repeat, bit_depth)


# Wrapper for batching environments together
class EnvBatcher():
    def __init__(self, env_class, env_args, env_kwargs, n):
        self.n = n
        self.envs = [env_class(*env_args, **env_kwargs) for _ in range(n)]
        self.dones = [True] * n

    # Resets every environment and returns observation
    def reset(self):
        observations = [env.reset() for env in self.envs]
        self.dones = [False] * self.n
        return torch.cat(observations)


# Steps/resets every environment and returns (observation, reward, done)

    def step(self, actions):
        done_mask = torch.nonzero(
            torch.tensor(self.dones)
        )[:,
          0]  # Done mask to blank out observations and zero rewards for previously terminated environments
        observations, rewards, dones = zip(
            *[env.step(action) if not d else (torch.zeros(1,3,128,128), 0,True) for env, action, d in zip(self.envs, actions, self.dones)])
        dones = [d or prev_d for d, prev_d in zip(dones, self.dones)
                 ]  # Env should remain terminated if previously terminated
        self.dones = dones
        observations, rewards, dones = torch.cat(observations), torch.tensor(
            rewards, dtype=torch.float32), torch.tensor(dones,
                                                        dtype=torch.uint8)
        observations[done_mask] = 0
        rewards[done_mask] = 0
        return observations, rewards, dones

    def close(self):
        [env.close() for env in self.envs]
