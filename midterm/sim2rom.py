import argparse
import torch
import cv2
from tetris import Tetris, SymbolTetrisSimple, rom2sim_id
from gym_tetris_simple.tetris_engine import PIECES, TEMPLATEWIDTH, TEMPLATEHEIGHT, BOARDWIDTH
from utils import str2bool
import gym
import numpy as np
import os
import gzip


parser = argparse.ArgumentParser("Implementation of Deep Q Network to play Tetris")
parser.add_argument("--width", type=int, default=10, help="The common width for all images")
parser.add_argument("--height", type=int, default=20, help="The common height for all images")
parser.add_argument("--block_size", type=int, default=5, help="Size of a block")
parser.add_argument("--total_episodes", type=int, default=1)
parser.add_argument("--fps", type=int, default=5, help="frames per second")
parser.add_argument("--saved_path", type=str, default="output")
parser.add_argument("--output", type=str, default="buffer_new.npz")
parser.add_argument("--out_video", type=str, default="video.avi")
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--gui_render", type=str2bool, default=True)
args = parser.parse_args()

def test(args):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)
    else:
        torch.manual_seed(0)
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
    model = torch.load("{}/tetris_5000".format(args.saved_path)).to(device)
    model.eval()

    rom_env = SymbolTetrisSimple(gym.make('Tetris-v0'), max_episode_length=-1)
    sim_env = Tetris(width=args.width, height=args.height, block_size=args.block_size)
    out = None
    if args.gui_render:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter("{}/{}".format(args.saved_path, args.out_video), fourcc, args.fps,
                                (args.width*args.block_size, args.height*args.block_size))
    episodes = 0
    while episodes < args.total_episodes:
        _, _, rom_done, _, _ = rom_env.reset(out)
        sim_env.reset()
        sim_env.update_board(rom_env._get_board_ram().astype(int))
        sim_env.set_new_piece(rom2sim_id(rom_env.game_state.fallingPiece))
        step_counter = 0
        while True:
            if (step_counter+1) % 500 == 0:
                print("Counter: %d Lines: %d" % (step_counter, sim_env.cleared_lines))
            next_steps = sim_env.get_next_states()
            next_actions, next_states = zip(*next_steps.items())
            next_states = torch.stack(next_states).to(device)
            predictions = model(next_states)[:, 0]
            index = torch.argmax(predictions).item()
            action = next_actions[index]
            
            _, done = sim_env.step(action, render=False)
            x, num_rotations = action
            # Note: current translation isn't optimal
            _, _, rom_done, _, _ = rom_env.step(num_rotations*10+x, video=out)
            done = rom_done or done
            if rom_env.game_state.fallingPiece is not None and not done:
                sim_env.update_board(rom_env._get_board_ram().astype(int))
                sim_env.set_new_piece(rom2sim_id(rom_env.game_state.fallingPiece))

            step_counter += 1
            if done:
                episodes += 1
                if args.gui_render:
                    out.release()
                break

        print("EpisodeID: %d\tSteps: %d\Lines: %d" % (episodes, step_counter, sim_env.cleared_lines))
    
if __name__ == "__main__":
    test(args)