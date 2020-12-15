import argparse
import os
import shutil
import random

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.tensorboard import SummaryWriter

from model import DQN
from tetris import Tetris
from utils import ReplayBufferOld
import utils

def get_args():
    parser = argparse.ArgumentParser(
        "Implementation of Deep Q Network to play Tetris")
    parser.add_argument("--width",type=int,default=10,
                        help="The common width for all images")
    parser.add_argument("--height",type=int,default=20,
                        help="The common height for all images")
    parser.add_argument("--block_size",type=int,default=30,
                        help="Size of a block")
    parser.add_argument("--batch_size",type=int,default=512,
                        help="The number of images per batch")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--initial_epsilon", type=float, default=1)
    parser.add_argument("--final_epsilon", type=float, default=1e-3)
    parser.add_argument("--num_decay_epochs", type=float, default=2000)
    parser.add_argument("--num_episodes", type=int, default=5000)
    parser.add_argument("--save_interval", type=int, default=500)
    parser.add_argument("--replay_memory_size",type=int,default=30000,
                        help="Number of epoches between testing phases")
    parser.add_argument("--log_path", type=str, default="runs")
    parser.add_argument("--saved_path", type=str, default="output")
    parser.add_argument("--gpu", type=int, default=1)

    args = parser.parse_args()
    return args


def get_epsilon(args, epoch):
    return args.final_epsilon + (max(args.num_decay_epochs - epoch, 0) *
                                 (args.initial_epsilon - args.final_epsilon) /
                                 args.num_decay_epochs)


def get_action_index(args, epoch, predictions, next_steps):
    epsilon = get_epsilon(args, epoch)
    if random.random() <= epsilon:
        action_index = random.randint(0, len(next_steps) - 1)
    else:
        action_index = torch.argmax(predictions).item()
    return action_index


def train(args):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)
    else:
        torch.manual_seed(0)
    if os.path.isdir(args.log_path):
        shutil.rmtree(args.log_path)
    os.makedirs(args.log_path)
    if not os.path.exists(args.saved_path):
        os.makedirs(args.saved_path)
    writer = SummaryWriter(args.log_path)
    env = Tetris(width=args.width,
                 height=args.height,
                 block_size=args.block_size)
    model = DQN()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    device = torch.device(
        'cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')

    state = env.reset()
    model.to(device)

    replay_memory = ReplayBufferOld(22, 2, device=device,
        max_size=args.replay_memory_size)  # action = [x_axis, rotate_times]
    episode = 0
    while episode < args.num_episodes:
        next_steps = env.get_next_states()

        next_actions, next_states = zip(*next_steps.items())
        next_states = torch.stack(next_states)
        next_states = next_states.to(device)
        model.eval()
        with torch.no_grad():
            predictions = model(next_states)[:, 0]
        index = get_action_index(args, episode, predictions, next_steps)
        model.train()

        next_state = next_states[index, :]
        next_state = next_state.cpu().numpy()
        action = next_actions[index]

        reward, done = env.step(action, render=False)

        replay_memory.add(state, action, next_state, reward, done)
        if done:
            final_score = env.score
            final_tetrominoes = env.tetrominoes
            final_cleared_lines = env.cleared_lines
            state = env.reset()
        else:
            state = next_state
            continue
        if len(replay_memory) < args.replay_memory_size:
            print("Current Memory Size: %d" % len(replay_memory))
            continue
        episode += 1
        batch = replay_memory.sample(args.batch_size)
        state_batch, action_batch, next_state_batch, reward_batch, done_batch = batch

        q_values = model(state_batch)
        model.eval()
        with torch.no_grad():
            next_prediction_batch = model(next_state_batch)
        model.train()

        next_prediction_batch[done_batch < 0.5] = 0.0
        y_batch = reward + next_prediction_batch

        optimizer.zero_grad()
        loss = criterion(q_values, y_batch)
        loss.backward()
        optimizer.step()

        print(
            "Episode: {}/{}, Action: {}, Score: {}, Tetrominoes {}, Cleared lines: {}"
            .format(episode, args.num_episodes, action, final_score,
                    final_tetrominoes, final_cleared_lines))
        writer.add_scalar('Train/Score', final_score, episode - 1)
        writer.add_scalar('Train/Tetrominoes', final_tetrominoes, episode - 1)
        writer.add_scalar('Train/Cleared lines', final_cleared_lines,
                          episode - 1)

        if episode > 0 and episode % args.save_interval == 0:
            torch.save(model, "{}/tetris_{}".format(args.saved_path, episode))

    torch.save(model, "{}/tetris".format(args.saved_path))


if __name__ == "__main__":
    args = get_args()
    train(args)
