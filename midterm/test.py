import argparse
import torch
import cv2
from tetris import Tetris
from utils import str2bool
import numpy as np

def get_args():
    parser = argparse.ArgumentParser("Implementation of Deep Q Network to play Tetris")

    parser.add_argument("--width", type=int, default=10, help="The common width for all images")
    parser.add_argument("--height", type=int, default=20, help="The common height for all images")
    parser.add_argument("--block_size", type=int, default=30, help="Size of a block")
    parser.add_argument("--fps", type=int, default=300, help="frames per second")
    parser.add_argument("--saved_path", type=str, default="output")
    parser.add_argument("--output", type=str, default="video.avi")
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--gui_render", type=str2bool, default=False)

    args = parser.parse_args()
    return args


def test(args):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)
    else:
        torch.manual_seed(0)
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
    model = torch.load("{}/tetris_5000".format(args.saved_path)).to(device)

    model.eval()
    env = Tetris(width=args.width, height=args.height, block_size=args.block_size)
    env.reset()
    out = None
    if args.gui_render:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter("{}/{}".format(args.saved_path, args.output), fourcc, args.fps,
                            (int(1.5*args.width*args.block_size), args.height*args.block_size))
    counter = 0
    while True:
        if counter % 100 == 0:
            print("Counter: %d Lines: %d" % (counter, env.cleared_lines))
        next_steps = env.get_next_states()
        next_actions, next_states = zip(*next_steps.items())
        next_states = torch.stack(next_states).to(device)
        predictions = model(next_states)[:, 0]
        index = torch.argmax(predictions).item()
        action = next_actions[index]
        _, done = env.step(action, render=args.gui_render, video=out)
        counter += 1
        if done:
            if args.gui_render:
                out.release()
            break
    return env.cleared_lines
        


if __name__ == "__main__":
    test_num = 10
    test_res = np.zeros(test_num)
    for i in range(test_num):
        print("Test Episode: %d" % i)
        args = get_args()
        test_res[i] = test(args)
    print(test_res, test_res.mean(), test_res.std())

