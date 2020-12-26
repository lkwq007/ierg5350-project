import numpy as np
import torch
from env import postprocess_observation, preprocess_observation_

import kornia
import torch
import torch.nn as nn
import torch.nn.functional as F


def preprocess_observation_(observation, bit_depth):
    observation.div_(2**(8 - bit_depth)).floor_().div_(2**bit_depth).sub_(
        0.5)  # Quantise to given bit depth and centre
    observation.add_(
        torch.rand_like(observation).div_(2**bit_depth)
    )  # Dequantise (to approx. match likelihood of PDF of continuous images vs. PMF of discrete images)
    return observation


# Postprocess an observation for storage (from float32 numpy array [-0.5, 0.5] to uint8 numpy array [0, 255])
def postprocess_observation(observation, bit_depth):
    return np.clip(
        np.floor((observation + 0.5) * 2**bit_depth) * 2**(8 - bit_depth), 0,
        2**8 - 1).astype(np.uint8)


# the API would change with drq, so I made a hard swtich

class ExperienceReplay():
    def __init__(self,
                 size,
                 observation_size,
                 action_size,
                 bit_depth,
                 device, image_pad=6):
        self.device = device
        self.size = size

        self.aug_trans = nn.Sequential(
            nn.ReplicationPad2d(image_pad),
            kornia.augmentation.RandomCrop((observation_size[-2], observation_size[-1])))

        self.observations = np.empty((size, *observation_size), dtype=np.uint8)
        # seems next_obs is also not suitable for dreamer
        # anyway, I just comment them out
        # self.next_observations = np.empty(
        #     (size, *observation_size), dtype=np.uint8)
        self.actions = np.empty((size, action_size), dtype=np.float32)
        self.rewards = np.empty((size, ), dtype=np.float32)
        self.nonterminals = np.empty((size, 1), dtype=np.float32)
        # seems useless
        self.not_dones_no_max = np.empty((size, 1), dtype=np.float32)

        self.idx = 0
        self.full = False  # Tracks if memory has been filled/all slots are valid
        # Tracks how much experience has been used in total
        self.steps, self.episodes = 0, 0
        self.bit_depth = bit_depth

    def append(self, observation, action, reward, done, next_observation=None):
        self.observations[self.idx] = postprocess_observation(
            observation.numpy(), self.bit_depth
        )  # Decentre and discretise visual observations (to save memory)
        # self.next_observations[self.idx] = postprocess_observation(
        #     next_observation.numpy(), self.bit_depth
        # )  # Decentre and discretise visual observations (to save memory)
        self.actions[self.idx] = action.numpy()
        self.rewards[self.idx] = reward
        self.nonterminals[self.idx] = not done
        # self.not_dones_no_max[self.idx] = not done_no_max
        self.idx = (self.idx + 1) % self.size
        self.full = self.full or self.idx == 0
        self.steps, self.episodes = self.steps + \
            1, self.episodes + (1 if done else 0)

    # Returns an index for a valid single sequence chunk uniformly sampled from the memory
    def _sample_idx(self, L):
        valid_idx = False
        while not valid_idx:
            idx = np.random.randint(
                0, self.size if self.full else self.idx - L)
            idxs = np.arange(idx, idx + L) % self.size
            # Make sure data does not cross the memory index
            valid_idx = not self.idx in idxs[1:]
        return idxs

    def _retrieve_batch(self, idxs, n, L):
        vec_idxs = idxs.transpose().reshape(-1)  # Unroll indices

        obs_ = self.observations[vec_idxs].astype(np.float32)
        # next_obs_ = self.next_observations[vec_idxs].astype(np.float32)
        # obs_aug = obs_.copy()
        # next_obs_aug = next_obs_.copy()

        # Undo discretisation for visual observations
        observations = torch.as_tensor(obs_)
        observations = preprocess_observation_(
            observations, self.bit_depth).to(self.device)

        # next_observations = torch.as_tensor(next_obs_)
        # next_observations = preprocess_observation_(
        #     next_observations, self.bit_depth).to(self.device)

        # I think we need to preserve the original observations for recontructions
        # observations0 = self.aug_trans(observations)
        # next_observations0 = self.aug_trans(next_observations)

        # observations_aug = observations.clone()
        observations_aug0 = self.aug_trans(observations)
        observations_aug1 = self.aug_trans(observations)

        # next_observations_aug = next_observations.clone()
        # next_observations_aug = self.aug_trans(next_observations_aug)

        return (observations.reshape(L, n, *observations.shape[1:]), self.actions[vec_idxs].reshape(L, n, -1), self.rewards[vec_idxs].reshape(L, n), self.nonterminals[vec_idxs].reshape(L, n, 1),
                observations_aug0.reshape(L, n, *observations_aug0.shape[1:]), observations_aug1.reshape(L, n, *observations_aug1.shape[1:]))

    # Returns a batch of sequence chunks uniformly sampled from the memory
    def sample(self, n, L):
        batch = self._retrieve_batch(np.asarray(
            [self._sample_idx(L) for _ in range(n)]), n, L)
        # print(np.asarray([self._sample_idx(L) for _ in range(n)]))
        # [1578 1579 1580 ... 1625 1626 1627]                                                                                                                                        | 0/100 [00:00<?, ?it/s]
        # [1049 1050 1051 ... 1096 1097 1098]
        # [1236 1237 1238 ... 1283 1284 1285]
        # ...
        # [2199 2200 2201 ... 2246 2247 2248]
        # [ 686  687  688 ...  733  734  735]
        # [1377 1378 1379 ... 1424 1425 1426]]
        return [torch.as_tensor(item).to(device=self.device) for item in batch]
