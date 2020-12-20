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
from utils import ReplayBufferOld, str2bool
import utils
import logging
import sys
import math

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
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--initial_epsilon", type=float, default=0.9)
    parser.add_argument("--final_epsilon", type=float, default=1e-3)
    parser.add_argument("--num_decay_epochs", type=float, default=1000)
    parser.add_argument("--num_episodes", type=int, default=100000)
    parser.add_argument("--max_episode_length", type=int, default=5000)
    parser.add_argument("--save_interval", type=int, default=200)
    parser.add_argument("--replay_memory_size",type=int,default=50000,
                        help="Number of epoches between testing phases")
    parser.add_argument("--tensorboard_dir", type=str, default="runs")
    parser.add_argument("--saved_dir", type=str, default="output")
    parser.add_argument("--log_file", type=str, default="output/train.log")
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--sim_rom_mode", type=str2bool, default=False)

    args = parser.parse_args()
    return args


def get_epsilon(args, epoch):
    # eps_threshold = args.final_epsilon + (max(args.num_decay_epochs - epoch, 0) * (
    #             args.initial_epsilon - args.final_epsilon) / args.num_decay_epochs)
    eps_threshold = args.final_epsilon + (args.initial_epsilon - args.final_epsilon) \
                    * math.exp(-1. * epoch / args.num_decay_epochs)
    return eps_threshold


def get_action_index(args, epoch, predictions, next_steps):
    epsilon = get_epsilon(args, epoch)
    if random.random() <= epsilon:
        action_index = random.randint(0, len(next_steps) - 1)
    else:
        action_index = torch.argmax(predictions).item()
    return action_index


def setup_logger(args):
    if not os.path.exists(os.path.dirname(args.log_file)):
        os.makedirs(os.path.dirname(args.log_file))
    logging.basicConfig(filename=args.log_file, filemode='w', format='%(message)s', level=logging.INFO)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)
    return logging.getLogger('myapp.area1')

def train(args):
    logger = setup_logger(args)
    logger.info('---- Options ----')
    for k, v in vars(args).items():
        logger.info(k + ': ' + str(v))
    logger.info('--------\n')
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)
    else:
        torch.manual_seed(0)
    if os.path.isdir(args.tensorboard_dir):
        shutil.rmtree(args.tensorboard_dir)
    os.makedirs(args.tensorboard_dir)
    if not os.path.exists(args.saved_dir):
        os.makedirs(args.saved_dir)
    writer = SummaryWriter(args.tensorboard_dir)
    env = Tetris(width=args.width,
                 height=args.height,
                 block_size=args.block_size,
                 sim_rom_mode=args.sim_rom_mode)
    state_dim = 25 
    action_dim = 2
    device = torch.device(
        'cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
    model = DQN(input_dim=state_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    state = env.reset()

    replay_memory = ReplayBufferOld(state_dim, action_dim, device=device,
        max_size=args.replay_memory_size)  # action = [x_axis, rotate_times]
    episode = 0
    step_cnt = 0
    seed = 0
    random.seed(seed)
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
        if step_cnt > args.max_episode_length:
            done = True

        replay_memory.add(state, action, next_state, reward, done)
        if done:
            final_score = env.score
            final_tetrominoes = env.tetrominoes
            final_cleared_lines = env.cleared_lines
            state = env.reset()
            step_cnt = 0
        else:
            state = next_state
            step_cnt += 1
            continue

        if len(replay_memory) < args.replay_memory_size / 10:
            # logger.info("Episode:%d Current Memory Size: %d" % (episode, len(replay_memory)))
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
        norm_lines_batch = torch.sqrt((reward_batch - 1) / 10) / 4
        y_batch = norm_lines_batch + args.gamma * next_prediction_batch
        # y_batch = reward_batch + args.gamma * next_prediction_batch

        optimizer.zero_grad()
        loss = criterion(q_values, y_batch)
        loss.backward()
        optimizer.step()

        logger.info(
            "Episode: {}/{}, Score: {}, Tetrominoes {}, Cleared lines: {}"
            .format(episode, args.num_episodes, final_score,
                    final_tetrominoes, final_cleared_lines))
        writer.add_scalar('Train/Score', final_score, episode - 1)
        writer.add_scalar('Train/Tetrominoes', final_tetrominoes, episode - 1)
        writer.add_scalar('Train/Cleared lines', final_cleared_lines,
                          episode - 1)

        if episode > 2000 and episode % args.save_interval == 0:
            torch.save(model, "{}/tetris_{}.pth".format(args.saved_dir, episode))
        if episode % 100:
            random.seed(seed%10)
            seed += 1

    torch.save(model, "{}/tetris.pth".format(args.saved_dir))


if __name__ == "__main__":
    args = get_args()
    train(args)
