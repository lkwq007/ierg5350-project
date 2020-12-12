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
parser.add_argument("--experience_size", type=int, default=1000000, help="Size of a block")
parser.add_argument("--fps", type=int, default=1, help="frames per second")
parser.add_argument("--saved_path", type=str, default="output")
parser.add_argument("--output", type=str, default="video.avi")
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--gui_render", type=str2bool, default=False)
args = parser.parse_args()

# global var
step_idx = 0 
finish_saving = False
all_reward = np.empty((args.experience_size, ), dtype=np.float32) 
all_action = np.empty((args.experience_size, ), dtype=np.int8) 
all_line = np.empty((args.experience_size, ), dtype=np.int8) 
all_done = np.empty((args.experience_size, ), dtype=np.bool) 
all_obs = np.empty((args.experience_size, 200), dtype=np.bool)

def get_onehot(act, action_space_size=6):
    one_hot_act = np.zeros(action_space_size, dtype=np.float32)
    one_hot_act[act] = 1
    return one_hot_act

def save_step(args, obs: np.ndarray, action: int, reward: float, done: bool, line: int):
    global step_idx, finish_saving, all_reward, all_action, all_done, all_obs, all_line
    all_obs[step_idx] = obs.astype(np.bool)
    all_action[step_idx] = action
    all_line[step_idx] = line
    all_reward[step_idx] = reward
    all_done[step_idx] = done
    if step_idx == args.experience_size - 1:
        finish_saving = True
    step_idx = step_idx + 1
    step_idx = step_idx % args.experience_size


# def wrapper(args, obs, action, reward, done):
#     obs = torch.FloatTensor(obs).unsqueeze(dim=0)
#     action = torch.from_numpy(get_onehot(action))
#     return obs, action, reward, done


def apply_sim_to_rom(
    args, 
    rom_env: SymbolTetrisSimple,
    obs: np.ndarray, 
    sim_action: tuple):
    piece_left_pos_x, num_rotations = sim_action
    flag = False
    done = False
    # rotate
    if rom_env.game_state.fallingPiece is not None and not done:
        act = 2
        for _ in range(num_rotations):
            next_obs, reward, done, _, line = rom_env.step(act) # up (rotate)
            save_step(args, obs, act, reward, done, line)
            obs = next_obs
            if rom_env.game_state.fallingPiece is None:
                break
            if done:
                break
    # move left/right
    if rom_env.game_state.fallingPiece is not None and not done:
        fallingPiece = rom_env.game_state.fallingPiece
        center_x_in_rom = 11
        init_x = int(BOARDWIDTH / 2) - int(TEMPLATEWIDTH / 2)
        for x in range(TEMPLATEWIDTH):
            for y in range(TEMPLATEHEIGHT):
                isAboveBoard = y + fallingPiece['y'] < 0
                if isAboveBoard or PIECES[fallingPiece['shape']][fallingPiece['rotation']][y][x] == '.':
                    continue
                center_x_in_rom = min(center_x_in_rom, x+init_x)
        offset = piece_left_pos_x - center_x_in_rom
        if offset != 0:
            act = 1 if offset < 0 else 3 # 1: left, 3: right
            for i in range(abs(offset)):
                next_obs, reward, done, _, line = rom_env.step(act) # move left or right
                save_step(args, obs, act, reward, done, line)
                obs = next_obs
                if rom_env.game_state.fallingPiece is None:
                    break
                if done:
                    break
    # down
    while rom_env.game_state.fallingPiece is not None and not done:
        act = 4
        next_obs, reward, done, _, line = rom_env.step(act) # down
        save_step(args, obs, act, reward, done, line)
        obs = next_obs
        if done:
            break
    return obs, done

def test(args):
    global step_idx, finish_saving, all_reward, all_action, all_done, all_obs, all_line
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)
    else:
        torch.manual_seed(0)
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
    model = torch.load("{}/tetris".format(args.saved_path)).to(device)
    model.eval()

    rom_env = SymbolTetrisSimple(gym.make('Tetris-v0'))
    env = Tetris(width=args.width, height=args.height, block_size=args.block_size)
    if args.gui_render:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter("{}/{}".format(args.saved_path, args.output), fourcc, args.fps,
                                (args.width*args.block_size, args.height*args.block_size))
    episodes = 0
    
    while not finish_saving:
        rom_obs = rom_env.reset()
        save_step(args, rom_obs, 0, 0.0, False, 0)
        env.reset()
        env.set_new_piece(rom2sim_id(rom_env.game_state.fallingPiece))
        counter = 0
        while True:
            if (counter+1) % 500 == 0:
                print("Counter: %d" % counter)
            next_steps = env.get_next_states()
            next_actions, next_states = zip(*next_steps.items())
            next_states = torch.stack(next_states).to(device)
            predictions = model(next_states)[:, 0]
            index = torch.argmax(predictions).item()
            action = next_actions[index]
            
            _, done = env.step(action, render=args.gui_render)
            # Note: current translation isn't optimal
            rom_res = apply_sim_to_rom(args, rom_env, rom_obs, action) 
            rom_obs, rom_done = rom_res
            done = rom_done or done
            if not done:
                env.update_board(rom_env._get_board_ram().astype(int))
                env.set_new_piece(rom2sim_id(rom_env.game_state.nextPiece))

            # skip fallingPiece is None (next obs: new piece will appear)
            next_rom_obs, reward, rom_done, _, line = rom_env.step(0) 
            save_step(args, rom_obs, 0, reward, rom_done, line)
            rom_obs = next_rom_obs
            done = rom_done or done
            
            # drawer
            if args.gui_render:
                img = rom_env.render(mode='rgb_array')
                out.write(cv2.flip(img, 1))

            counter += 1
            if done:
                episodes += 1
                if args.gui_render:
                    out.release()
                break

        print("Step: %d\tScore: %d\tSteps: %d\tEpisodes: %d" % (counter, env.score, step_idx, episodes))
    
    print("Saving buffer...")
    rom_env.close()
    np.savez(os.path.join("output", "buffer.npz"), 
        all_reward=all_reward, 
        all_action=all_action, 
        all_line=all_line, 
        all_done=all_done, 
        all_obs=all_obs
    )

if __name__ == "__main__":
    test(args)