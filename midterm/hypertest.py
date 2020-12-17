import argparse
import torch
import gym
from model import SimpleTetrisDQN, SimpleTetrisConvDQN
from tetris import SymbolTetrisSimple
import cv2
from utils import str2bool

def get_args():
    parser = argparse.ArgumentParser("Implementation of Deep Q Network to play Tetris")

    parser.add_argument("--width", type=int, default=10, help="The common width for all images")
    parser.add_argument("--height", type=int, default=20, help="The common height for all images")
    parser.add_argument("--block_size", type=int, default=5, help="Size of a block")
    parser.add_argument("--fps", type=int, default=10, help="frames per second")
    parser.add_argument("--saved_path", type=str, default="output")
    parser.add_argument("--ckpt_name", type=str, default="tetris_18000.pth")
    parser.add_argument("--out_video", type=str, default="video.avi")
    parser.add_argument("--sample_img", type=str, default="output.png")
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--gui_render", type=str2bool, default=True)

    args = parser.parse_args()
    return args


def test(args):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)
    else:
        torch.manual_seed(0)
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
    out = None
    if args.gui_render:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter("{}/{}".format(args.saved_path, args.out_video), fourcc, args.fps,
                                (args.width*args.block_size, args.height*args.block_size))
    
    state_dicts = torch.load("{}/{}".format(args.saved_path, args.ckpt_name))
    env = SymbolTetrisSimple(gym.make('Tetris-v0'), 
                            max_episode_length=-1, align=False)
    n_actions = env.game_board_width * 4 # board_width * rotation
    state_size = env.game_board_width * env.game_board_height
    info_size = 2 * env.num_tetris_kind

    policy_net = SimpleTetrisConvDQN(state_size=state_size, info_size=info_size).to(device)
    policy_net.load_state_dict(state_dicts['policy_net'])
    policy_net.to(device)
    policy_net.eval()
    
    state, reward, done, info, line = env.reset(video=out)
    
    state = state.reshape(-1)
    total_steps = 0
    total_reward = 0.0
    total_line = 0
    while True:
        if total_steps % 100 == 0:
            print("Total Steps: %d" % total_steps)
        state = torch.FloatTensor(state).to(device).unsqueeze(0)
        info = torch.FloatTensor(info).to(device).unsqueeze(0)
        action = policy_net(state, info).max(1)[1].view(1, 1).item()
        next_state, reward, done, next_info, line = env.step(action, video=out)
        state = next_state.reshape(-1)
        info = next_info
        total_reward += reward
        total_line += line
        total_steps += 1
        if done:
            if out is not None:
                out.release()
            break
        if not done:
            cv2.imwrite("{}/{}".format(args.saved_path, args.sample_img), 
                        cv2.flip(env.render(mode='rgb_array'), 1))
    
    
    print("Steps: %d Reward %.4f Cleared lines: %d" % (total_steps, total_reward, round(total_line)))


if __name__ == "__main__":
    args = get_args()
    test(args)
