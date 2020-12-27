import argparse
import os
import shutil
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import gym
from itertools import count
from model import SimpleTetrisDQN, SimpleTetrisConvDQN
from tetris import SymbolTetrisSimple, rom2sim_id, Tetris
from utils import ReplayBuffer, str2bool, load_dataset


parser = argparse.ArgumentParser("Implementation of Deep Q Network to play Tetris")
parser.add_argument("--block_size", type=int, default=30, help="Size of a block")
parser.add_argument("--batch_size", type=int, 
                default=1024, help="The number of images per batch")
parser.add_argument("--lr", type=float, default=5e-4)
parser.add_argument("--gamma", type=float, default=0.999)
parser.add_argument("--target_update", type=int, default=10)
parser.add_argument("--initial_epsilon", type=float, default=0.05)
parser.add_argument("--final_epsilon", type=float, default=0.001)
parser.add_argument("--eps_decay", type=float, default=10000)
parser.add_argument("--num_episodes", type=int, default=1000000)
parser.add_argument("--max_episode_length", type=int, default=3000)
parser.add_argument("--save_interval", type=int, default=5000)
parser.add_argument("--replay_memory_size", type=int, default=1e7, 
                help="Number of epoches between testing phases")
parser.add_argument("--log_path", type=str, default="runs")
parser.add_argument("--saved_path", type=str, default="output")
parser.add_argument("--imitation_data_path", type=str, default="output/buffer_imitation.npz")
parser.add_argument("--imitation_episodes", type=int, default=1000000)
parser.add_argument("--gpu", type=int, default=0)

args = parser.parse_args()


BATCH_SIZE = args.batch_size
GAMMA = args.gamma
TARGET_UPDATE = args.target_update
EPS_START = args.initial_epsilon
EPS_END = args.final_epsilon
EPS_DECAY = args.eps_decay 
imitation_loss_fn = nn.CrossEntropyLoss()

if torch.cuda.is_available():
    torch.cuda.manual_seed(0)
else:
    torch.manual_seed(0)
# if os.path.isdir(args.log_path):
#     shutil.rmtree(args.log_path)
# os.makedirs(args.log_path)
if not os.path.exists(args.saved_path):
    os.makedirs(args.saved_path)
writer = SummaryWriter(args.log_path)
device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")

env = SymbolTetrisSimple(gym.make('Tetris-v0'), 
                        max_episode_length=args.max_episode_length, align=False)
n_actions = env.game_board_width * 4 # board_width * rotation
state_size = env.game_board_width * env.game_board_height
info_size = 2 * env.num_tetris_kind

policy_net = SimpleTetrisConvDQN(state_size=state_size, info_size=info_size).to(device)
target_net = SimpleTetrisConvDQN(state_size=state_size, info_size=info_size).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
optimizer = torch.optim.Adam(policy_net.parameters(), lr=args.lr)
criterion = nn.MSELoss()

memory = ReplayBuffer(state_size, 1, info_size, device=device, max_size=args.replay_memory_size) 
memory = load_dataset(memory, args.imitation_data_path) # for imitation learning

def select_action(state, info, i_episode):
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * i_episode / EPS_DECAY)
    if sample > eps_threshold:
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device).unsqueeze(0)
            info = torch.FloatTensor(info).to(device).unsqueeze(0)
            return policy_net(state, info).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


def optimize_model():
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = memory.sample(BATCH_SIZE)
    state_batch, action_batch, next_state_batch, reward_batch, not_done_batch, info_batch, next_info_batch = batch

    state_action_values = policy_net(state_batch, info_batch).gather(1, action_batch)

    next_state_values = torch.zeros((BATCH_SIZE, 1), device=device)
    mask = (not_done_batch == 1).squeeze(1)
    next_state_values[mask] = target_net(next_state_batch[mask], next_info_batch[mask]).max(1, keepdim=True)[0].detach()

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    loss = F.mse_loss(state_action_values, expected_state_action_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


def imitation_learning():
    batch = memory.sample(BATCH_SIZE)
    state_batch, action_batch, next_state_batch, reward_batch, not_done_batch, info_batch, next_info_batch = batch

    state_action_values = policy_net(state_batch, info_batch)

    loss = imitation_loss_fn(state_action_values, action_batch.reshape(-1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # for param in policy_net.parameters():
    #     param.grad.data.clamp_(-1, 1)
    optimizer.step()
    return loss.item()

# imitation learning
total_loss = 0.0
for i in range(args.imitation_episodes):
    loss = imitation_learning()
    total_loss += loss
    avg_loss = total_loss / (i+1)
    if i % 100 == 0:
        print("Episodes: %d/%d\tPolicy Loss: %.4f/Avg Loss: %.4f" % 
            (i, args.imitation_episodes, loss, avg_loss))
target_net.load_state_dict(policy_net.state_dict())
print("Finish imitation learning")

episode_durations = []
all_steps = []
all_reward = []
all_lines = []

for i_episode in range(args.num_episodes):       
    next_state, reward, done, next_info, line = env.reset()
    state = next_state.reshape(-1)
    info = next_info
    ep_reward = 0.0
    ep_line = 0
    ep_steps = 0
    for t in count():
        action = select_action(state, info, i_episode).item()
        next_state, reward, done, next_info, line = env.step(action)
        next_state = next_state.reshape(-1)
        if done:
            next_state = None
        memory.add(state, action, next_state, reward, done, info, next_info)
        state = next_state
        info = next_info
        
        optimize_model()

        ep_reward += reward
        ep_line += line
        ep_steps = t

        if done:
            episode_durations.append(t + 1)
            break
    
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    all_steps.append(ep_steps)
    all_reward.append(ep_reward)
    all_lines.append(ep_line)
    
    if i_episode % 50 == 0:
        print(
            "Episodes: %d/%d\tSteps: %d/%.4f\tReward: %.4f/%.4f\tLines: %d/%d" % 
            (i_episode, args.num_episodes, 
            all_steps[-1], sum(all_steps)/len(all_steps),
            all_reward[-1], sum(all_reward)/len(all_reward),
            round(all_lines[-1]), sum(all_lines)
            )
        )
    writer.add_scalar("Train/Reward", all_reward[-1], i_episode - 1)
    writer.add_scalar("Train/Steps", all_steps[-1], i_episode - 1)
    writer.add_scalar("Train/Lines", all_lines[-1], i_episode - 1)

    if i_episode > 0 and i_episode % args.save_interval == 0:
        state_dict = {
            'policy_net': policy_net.state_dict(), 
            'optimizer': optimizer.state_dict()
        }
        torch.save(state_dict, "{}/tetris_{}.pth".format(args.saved_path, i_episode))

state_dict = {
    'policy_net': policy_net.state_dict(), 
    'optimizer': optimizer.state_dict()
}
torch.save(state_dict, "{}/tetris_final.pth".format(args.saved_path))

