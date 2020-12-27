import argparse
import torch
import cv2
from tetris_eval import Tetris
from utils import str2bool
import numpy as np
import random

def get_args():
    parser = argparse.ArgumentParser("Implementation of Deep Q Network to play Tetris")

    parser.add_argument("--width", type=int, default=10, help="The common width for all images")
    parser.add_argument("--height", type=int, default=20, help="The common height for all images")
    parser.add_argument("--block_size", type=int, default=30, help="Size of a block")
    parser.add_argument("--fps", type=int, default=300, help="frames per second")
    parser.add_argument("--saved_path", type=str, default="output")
    parser.add_argument("--ckpt_name", type=str, default="tetris_24500.pth")
    parser.add_argument("--output", type=str, default="video.avi")
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--gui_render", type=str2bool, default=False)
    parser.add_argument("--agent", type=str, default="dqn",choices=["dqn","1","2","3","dqnstop"])
    args = parser.parse_args()
    return args


def test(args,seed=0):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)
    else:
        torch.manual_seed(0)
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
    model = torch.load("{}/{}".format(args.saved_path, args.ckpt_name), map_location=device)
    random.seed(seed)
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
            if env.cleared_lines>=10000 and args.agent=='dqnstop':
                return env.cleared_lines
        next_steps, next_boards = env.get_next_states()
        next_actions, next_states = zip(*next_steps.items())
        if args.agent!='dqn' and args.agent!='dqnstop':
            next_actions, boards = zip(*next_boards.items())
            index=0
            max_score=None
            for i in range(len(boards)):
                board, pos_y, cleared, new_board = boards[i]
                # board=new_board
                # print(board)
                # exit(0)
                board[board>0.1]=1.0
                board=board.astype(np.float32)
                # cleared = next_states[next_actions[i]][-2].item()
                if args.agent=="3":
                    bumpiness, heights, holes, max_heights = get_bumpiness_height_hole2(board)
                else:
                    bumpiness, heights, holes, max_heights = get_bumpiness_height_hole(board)
                landing_height=20-pos_y+1
                row_t,col_t = get_cell_changes(board)
                wells = get_wells(board)
                # print(board)
                # print(landing_height)
                # print(row_t)
                # print(col_t)
                # print(wells)
                # exit(0)
                # print(board.shape)
                if args.agent=="1" or args.agent=="3":
                    score = -0.51*heights+0.76*cleared-0.36*holes-0.18*bumpiness
                else:
                    score=0
                    score = -4.50*landing_height+3.42*cleared \
                        -3.22* row_t.sum() -9.35*col_t.sum() \
                        - 7.90*holes - 3.39*wells
                    score = -4.500158825082766*landing_height+3.4181268101392694*cleared \
                -3.2178882868487753* row_t.sum() -9.348695305445199*col_t.sum() \
                - 7.899265427351652*holes - 3.3855972247263626*wells
                if max_score is None or score>max_score:
                    index=i
                    max_score=score
        else:
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
        

def get_bumpiness_height_hole(board):
    board = np.array(board)
    mask = board > 0
    invert_heights = np.where(mask.any(axis=0), np.argmax(mask, axis=0), 20)
    heights = 20 - invert_heights
    total_height = np.sum(heights)
    max_heights=heights.max()
    currs = heights[:-1]
    nexts = heights[1:]
    diffs = np.abs(currs - nexts)
    total_bumpiness = np.sum(diffs)
    total_cell=np.sum(board)
    # print(total_cell)
    # exit(0)
    return total_bumpiness, total_height, total_height-total_cell, max_heights

def get_bumpiness_height_hole2(board):
    board = np.array(board)
    mask = board > 0
    invert_heights = np.where(mask.any(axis=0), np.argmax(mask, axis=0), 20)
    heights = 20 - invert_heights
    total_height = np.sum(heights)
    max_heights=heights.max()
    currs = heights[:-1]
    nexts = heights[1:]
    diffs = np.abs(currs - nexts)
    total_bumpiness = np.sum(diffs)+heights[0]+heights[-1]
    total_cell=np.sum(board)
    # print(total_cell)
    # exit(0)
    return total_bumpiness, total_height, total_height-total_cell, max_heights

def get_height_vector(board):
    board = np.array(board)
    mask = board > 0
    invert_heights = np.where(mask.any(axis=0), np.argmax(mask, axis=0), 20)
    heights = boards.shape[0] - invert_heights
    total = np.sum(np.minimum(board,1), axis=0)
    return heights,heights - total

def get_cell_changes(board):
    # Row Transitions, Column Transitions
    return abs(np.diff(board, axis=1)).sum(axis=1),abs(np.diff(board, axis=0)).sum(axis=0)

def get_wells(board):
    x=board
    y=x[:,1:-1]
    y1=x[:,0:-2]
    y2=x[:,2:]
    z=np.bitwise_and((y1+y2+y==2),y==0)
    ret=0
    for i in range(8):
        acc=0
        cnt=0
        for j in range(board.shape[0]):
            if z[j][i]:
                cnt+=1
                acc+=cnt
            else:
                cnt=0
        ret+=acc
    for i in range(board.shape[0]):
        acc=0
        cnt=0
        
    return ret


if __name__ == "__main__":
    test_num = 100
    test_res = np.zeros(test_num)
    # random.seed(1)
    for i in range(test_num):
        print("Test Episode: %d" % i)
        args = get_args()
        test_res[i] = test(args, i)
    print(test_res, test_res.mean(), test_res.std())
    np.save(f"{args.agent}.npy",test_res)
    with open(f"{args.agent}.log","w") as f:
        print(test_res, test_res.mean(), test_res.std(), file=f)

