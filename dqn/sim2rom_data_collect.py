import argparse
import torch
import cv2
from tetris import Tetris, SymbolTetrisSimple, rom2sim_id
from gym_tetris_simple.tetris_engine import PIECES, TEMPLATEWIDTH, TEMPLATEHEIGHT, BOARDWIDTH
from utils import str2bool, ReplayBuffer
import gym
import numpy as np
import os
import gzip


parser = argparse.ArgumentParser("Implementation of Deep Q Network to play Tetris")
parser.add_argument("--width", type=int, default=10, help="The common width for all images")
parser.add_argument("--height", type=int, default=20, help="The common height for all images")
parser.add_argument("--block_size", type=int, default=5, help="Size of a block")
parser.add_argument("--experience_size", type=int, default=1000000, help="Size of the buffer")
parser.add_argument("--total_episodes", type=int, default=10)
parser.add_argument("--max_episode_length", type=int, default=3000)
parser.add_argument("--fps", type=int, default=5, help="frames per second")
parser.add_argument("--saved_path", type=str, default="output")
parser.add_argument("--ckpt_name", type=str, default="tetris_24500.pth")
parser.add_argument("--output", type=str, default="buffer_imitation.npz")
parser.add_argument("--out_video", type=str, default="video.avi")
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--gui_render", type=str2bool, default=True)
args = parser.parse_args()

# global var
step_idx = 0 
finish_saving = False

class ImitationDataset(object):
    def __init__(self, buffer_size, state_size, info_size) -> None:
        super().__init__()
        self.ptr = 0 
        self.is_finish = False
        self.buffer_size = buffer_size
        self.all_state = np.empty((args.experience_size, state_size), dtype=np.bool)
        self.all_action = np.empty((args.experience_size, ), dtype=np.int8) 
        self.all_next_state = np.empty((args.experience_size, state_size), dtype=np.bool)
        self.all_reward = np.empty((args.experience_size, ), dtype=np.float32) 
        self.all_done = np.empty((args.experience_size, ), dtype=np.bool) 
        self.all_info = np.empty((args.experience_size, info_size*2), dtype=np.bool) 
        self.all_next_info = np.empty((args.experience_size, info_size*2), dtype=np.bool) 
    
    def add(self, state, action, next_state, reward, done, info, next_info):
        self.all_state[self.ptr] = state
        self.all_action[self.ptr] = action
        self.all_next_state[self.ptr] = next_state
        self.all_reward[self.ptr] = reward
        self.all_done[self.ptr] = done
        self.all_info[self.ptr] = info
        self.all_next_info[self.ptr] = next_info
        if self.ptr == self.buffer_size - 1:
            self.is_finish = True
        self.ptr = (self.ptr + 1) % self.buffer_size 
    
    def save(self, npz_save_path):
        np.savez(npz_save_path, 
            all_state=self.all_state, 
            all_action=self.all_action, 
            all_next_state=self.all_next_state, 
            all_reward=self.all_reward, 
            all_done=self.all_done, 
            all_info=self.all_info,
            all_next_info=self.all_next_info
        )
        print("Save buffer in %s" % npz_save_path)



def test(args):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)
    else:
        torch.manual_seed(0)
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
    model = torch.load("{}/{}".format(args.saved_path, args.ckpt_name), map_location=device)
    model.eval()

    rom_env = SymbolTetrisSimple(gym.make('Tetris-v0'), max_episode_length=args.max_episode_length, align=False)
    sim_env = Tetris(width=args.width, height=args.height, block_size=args.block_size)
    memory = ImitationDataset(buffer_size=args.experience_size, 
                        state_size=rom_env.observation_size, info_size=rom_env.num_tetris_kind)

    out = None
    if args.gui_render:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter("{}/{}".format(args.saved_path, args.out_video), fourcc, args.fps,
                                (args.width*args.block_size, args.height*args.block_size))
    
    episodes = 0
    while not memory.is_finish:
        rom_next_state, rom_reward, rom_done, rom_next_info, rom_line = rom_env.reset(out)
        rom_state = rom_next_state.reshape(-1)
        rom_info = rom_next_info

        sim_env.reset()
        sim_env.update_board(rom_env._get_board_ram().astype(int))
        sim_env.set_new_piece(rom2sim_id(rom_env.game_state.fallingPiece))
        step_counter = 0
        while True:
            sim_next_steps = sim_env.get_next_states()
            sim_next_actions, sim_next_states = zip(*sim_next_steps.items())
            sim_next_states = torch.stack(sim_next_states).to(device)
            sim_predictions = model(sim_next_states)[:, 0]
            index = torch.argmax(sim_predictions).item()
            sim_action = sim_next_actions[index]
            _, done = sim_env.step(sim_action, render=False)
            x, num_rotations = sim_action

            rom_action = num_rotations*10+x
            rom_next_state, rom_reward, rom_done, rom_next_info, rom_line = rom_env.step(rom_action, video=out)
            rom_next_state = rom_next_state.reshape(-1)
            done = rom_done or done
            if rom_env.game_state.fallingPiece is not None and not done:
                sim_env.update_board(rom_env._get_board_ram().astype(int))
                sim_env.set_new_piece(rom2sim_id(rom_env.game_state.fallingPiece))

            step_counter += 1
            if done:
                rom_next_state = None
            memory.add(rom_state, rom_action, rom_next_state, rom_reward, done, rom_info, rom_next_info)
            rom_state = rom_next_state
            rom_info = rom_next_info

            if done:
                episodes +=1
                if args.gui_render:
                    out.release()
                break

        print("Total_step: %d\tEpisodeID: %d\tSteps: %d\tLines: %d" % 
            (memory.ptr, episodes, step_counter, sim_env.cleared_lines))
    memory.save(os.path.join(args.saved_path, args.output))
    
if __name__ == "__main__":
    test(args)
    