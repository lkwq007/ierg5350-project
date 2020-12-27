import numpy as np
import torch
import random
from tqdm import tqdm

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, info_dim, device=None, max_size=int(1e6)):
		self.max_size = int(max_size)
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((self.max_size, state_dim), dtype=np.bool)
		self.action = np.zeros((self.max_size, action_dim), dtype=np.int8)
		self.next_state = np.zeros((self.max_size, state_dim), dtype=np.bool)
		self.reward = np.zeros((self.max_size, 1), dtype=np.float32)
		self.not_done = np.zeros((self.max_size, 1), dtype=np.bool)
		self.info = np.zeros((self.max_size, info_dim), dtype=np.bool)
		self.next_info = np.zeros((self.max_size, info_dim), dtype=np.bool)

		self.device = device


	def add(self, state, action, next_state, reward, done, info, next_info):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1 - done
		self.info[self.ptr] = info
		self.next_info[self.ptr] = next_info

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)

	def __len__(self):
		return self.size

	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)

		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.LongTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device),
			torch.FloatTensor(self.info[ind]).to(self.device),
			torch.FloatTensor(self.next_info[ind]).to(self.device)
		)


class ReplayBufferOld(object):
	def __init__(self, state_dim, action_dim, device=None, max_size=int(1e6)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))

		if device is None:
			self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		else:
			self.device = device


	def add(self, state, action, next_state, reward, done):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)

	def __len__(self):
		return self.size

	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)

		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device)
		)


def load_dataset(memory: ReplayBuffer, dataset_path:str):
    raw_data = np.load(dataset_path)
    all_state = raw_data['all_state'].copy()
    all_action = raw_data['all_action'].copy()
    all_next_state = raw_data['all_next_state'].copy()
    all_reward = raw_data['all_reward'].copy()
    all_done = raw_data['all_done'].copy()
    all_info = raw_data['all_info'].copy()
    all_next_info = raw_data['all_next_info'].copy()
    total_size = raw_data["all_reward"].shape[0]
    print("Loading Memory...")
    for i in tqdm(range(total_size)):
        memory.add(all_state[i], all_action[i], 
                all_next_state[i], all_reward[i], 
                all_done[i], all_info[i], all_next_info[i])
    return memory