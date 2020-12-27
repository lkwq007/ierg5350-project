"""

original author: Viet Nguyen <nhviet1009@gmail.com>
url: https://github.com/uvipen/Tetris-deep-Q-learning-pytorch/blob/master/src/tetris.py

"""
import numpy as np
from PIL import Image
import cv2
from matplotlib import style
import torch
import random
import gym_tetris_simple
import gym
from gym import spaces
from gym_tetris_simple.tetris_engine import PIECES, TEMPLATEWIDTH, TEMPLATEHEIGHT, BOARDWIDTH
import math
import torch.nn.functional as F

style.use("ggplot")

SIMPLE_MOVEMENT = [
    ['NOOP'],
    ['A'],
    ['B'],
    ['right'],
    ['left'],
    ['down'],
]

LUT_FOR_ROM_PIECE_2SIM_PIECE_ID = {
    'O': {0:1},
    'T': {0:2, 1:10, 2:8, 3:9},
    'S': {0:3, 1:11},
    'Z': {0:4, 1:12},
    'I': {0:5, 1:13},
    'L': {0:6, 1:15, 2:14, 3:16},
    'J': {0:7, 1:18, 2:17, 3:19},
}

LUT_FOR_ROM_PIECE_2PIECE_ID = {
    'O': 0,
    'T': 1,
    'S': 2,
    'Z': 3,
    'I': 4,
    'L': 5,
    'J': 6,
}

def rom2sim_id(rom_piece):
    shape = rom_piece['shape']
    rotation = rom_piece['rotation']
    # Note: SIM_PIECE_ID start from 1 in the Look up table
    # However, Tetris.pieces[0] == [[1, 1],[1, 1]]
    # We need to minus the offset 1 between the list and the Look up table
    ind = LUT_FOR_ROM_PIECE_2SIM_PIECE_ID[shape][rotation] - 1
    return  ind

def crop_image_simple(image):
    '''
    Args: image as an numpy array
    returns: a crop image as a standardised numpy array
    '''
    image = np.mean(image, axis=2)
    image[image > 0] = 1
    image = cv2.resize(image, (10,20))
    image = image.astype(np.float32)
    return image

def get_bumpiness_height_hole(board):
    board = np.array(board)
    mask = board > 0
    invert_heights = np.where(mask.any(axis=0), np.argmax(mask, axis=0), 20)
    heights = 20 - invert_heights
    total_height = np.sum(heights)
    max_height = heights.max()
    currs = heights[:-1]
    nexts = heights[1:]
    diffs = np.abs(currs - nexts)
    total_bumpiness = np.sum(diffs)
    total_cell=np.sum(board)
    return total_bumpiness, total_height, total_height-total_cell, max_height

class SymbolTetrisSimple(gym.Wrapper):
    def __init__(self, env, add_reward=False, old_reward=True, max_episode_length=100, align=True):
        gym.Wrapper.__init__(self, env)
        shp = env.observation_space.shape
        # set 
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(200,), dtype=np.float32)
        self.die = -50
        self.score = 0.0
        self.add_reward = add_reward
        self.old_reward = old_reward
        self.max_episode_length = max_episode_length
        self.align = align
        self.t = 0
        if not self.add_reward:
            self.die = 0

    def reset(self, video=None):
        obs = self.env.reset()
        self.score = 0.0
        self.t = 0

        fallingPieceID = 0 # fallingPiece is None
        nextPieceID = 0 # nextPiece is None
        done = False
        total_reward = 0.0
        total_cleared = 0
        # align fallingPiece
        if self.align:
            while self.env.game_state.fallingPiece is not None and not done:
                if self.env.game_state.fallingPiece['rotation'] == 0:
                    break
                if self.env.game_state.fallingPiece['rotation'] == 1:
                    next_obs, reward, done, _, cleared = self.step_kernel(5, video) # rotate
                    total_reward += reward
                    total_cleared += cleared
                    obs = next_obs
                else:
                    next_obs, reward, done, _, cleared = self.step_kernel(2, video) # rotate
                    total_reward += reward
                    total_cleared += cleared
                    obs = next_obs
                if done:
                    break
        if self.env.game_state.fallingPiece is not None and not done:
            fallingPieceID = LUT_FOR_ROM_PIECE_2PIECE_ID[self.env.game_state.fallingPiece['shape']]
        if self.env.game_state.nextPiece is not None and not done:
            nextPieceID = LUT_FOR_ROM_PIECE_2PIECE_ID[self.env.game_state.nextPiece['shape']]
        info = self.get_onehot((fallingPieceID, nextPieceID))
        # return self._get_board(obs), reward, done, info, cleared
        return self._get_board_ram(), total_reward, done, info, total_cleared # TODO: use ram board only
    
    def _get_board(self,obs):
        return crop_image_simple(obs).reshape(-1)
    
    def _get_board_ram(self):
        board=np.array(self.env.game_state.board)
        board[board!="."]=1
        board[board=="."]=0
        board=board.astype(np.float32).transpose(1,0)
        return board

    def step_kernel(self, action, video=None):            
        obs, reward, done, _ = self.env.step(action)
        if video is not None:
            video.write(cv2.flip(self.env.render(mode='rgb_array'), 1))
        cleared = round(reward / 100)
        
        if self.old_reward:
            score = math.pow(cleared, 2) * 10 # cleared^2 * width 
            self.score += score
            if done:
                reward -= 2
                self.score -=2
            reward = score
        # reward shaping
        if self.add_reward:
            board=self._get_board_ram()
            bumpiness, heights, holes, max_heights = get_bumpiness_height_hole(board)
            score = 4.76*cleared - 0.51*heights - 0.36*holes - 0.18*bumpiness
            reward = score - self.score
            self.score = score
        
        return self._get_board(obs), reward, done, None, cleared

    def step(self, sim_action, video=None):
        piece_left_pos_x = sim_action % 10
        num_rotations = sim_action // 10
        flag = False
        done = False
        total_cleared = 0
        if self.old_reward:
            total_reward = 1.0 # living reward
        else:
            total_reward = 0.0
        # rotate
        if self.env.game_state.fallingPiece is not None and not done:
            if num_rotations == 1:
                next_obs, reward, done, _, cleared = self.step_kernel(2, video) # rotate 90
                total_reward += reward
                total_cleared += cleared
                obs = next_obs
            elif num_rotations == 2:
                for _ in range(2):
                    next_obs, reward, done, _, cleared = self.step_kernel(2, video) # rotate 180
                    total_reward += reward
                    total_cleared += cleared
                    obs = next_obs
                    if self.env.game_state.fallingPiece is None:
                        break
                    if done:
                        break
            elif num_rotations == 3:
                next_obs, reward, done, _, cleared = self.step_kernel(5, video) # rotate -90
                total_reward += reward
                total_cleared += cleared
                obs = next_obs
        # move left/right
        if self.env.game_state.fallingPiece is not None and not done:
            fallingPiece = self.env.game_state.fallingPiece
            center_x_in_rom = 11
            init_x = int(BOARDWIDTH / 2) - int(TEMPLATEWIDTH / 2)
            for x in range(TEMPLATEWIDTH):
                for y in range(TEMPLATEHEIGHT):
                    # isAboveBoard = y + fallingPiece['y'] < 0
                    # if isAboveBoard or PIECES[fallingPiece['shape']][fallingPiece['rotation']][y][x] == '.':
                    if PIECES[fallingPiece['shape']][fallingPiece['rotation']][y][x] == '.':
                        continue
                    center_x_in_rom = min(center_x_in_rom, x+init_x)
            offset = piece_left_pos_x - center_x_in_rom
            if offset != 0:
                act = 1 if offset < 0 else 3 # 1: left, 3: right
                for i in range(abs(offset)):
                    next_obs, reward, done, _, cleared = self.step_kernel(act, video) # move left or right
                    total_reward += reward
                    total_cleared += cleared
                    obs = next_obs
                    if self.env.game_state.fallingPiece is None:
                        break
                    if done:
                        break
        # down
        while self.env.game_state.fallingPiece is not None and not done:
            act = 4
            next_obs, reward, done, _, cleared = self.step_kernel(act, video) # down
            total_reward += reward
            total_cleared += cleared
            obs = next_obs
            if done:
                break
        # get next piece
        fallingPieceID = 0 # fallingPiece is None
        nextPieceID = 0 # nextPiece is None
        if not done:
            next_obs, reward, done, _, cleared = self.step_kernel(0, video) # none
            total_reward += reward
            total_cleared += cleared
            obs = next_obs
            # align fallingPiece
            if self.align:
                while self.env.game_state.fallingPiece is not None and not done:
                    if self.env.game_state.fallingPiece['rotation'] == 0:
                        break
                    if self.env.game_state.fallingPiece['rotation'] == 1:
                        next_obs, reward, done, _, cleared = self.step_kernel(5, video) # rotate
                        total_reward += reward
                        total_cleared += cleared
                        obs = next_obs
                    else:
                        next_obs, reward, done, _, cleared = self.step_kernel(2, video) # rotate
                        total_reward += reward
                        total_cleared += cleared
                        obs = next_obs
                    if done:
                        break
            if self.env.game_state.fallingPiece is not None and not done:
                fallingPieceID = LUT_FOR_ROM_PIECE_2PIECE_ID[self.env.game_state.fallingPiece['shape']]
            if self.env.game_state.nextPiece is not None and not done:
                nextPieceID = LUT_FOR_ROM_PIECE_2PIECE_ID[self.env.game_state.nextPiece['shape']]
        info = self.get_onehot((fallingPieceID, nextPieceID))
        self.t += 1
        done = done or self.t == self.max_episode_length
        return self._get_board_ram(), total_reward, done, info, total_cleared
    
    def get_onehot(self, info):
        fall_info = np.zeros(self.num_tetris_kind, dtype=np.float32)
        fall_info[info[0]] = 1
        next_info = np.zeros(self.num_tetris_kind, dtype=np.float32)
        next_info[info[1]] = 1
        return np.concatenate((fall_info, next_info), axis=None)
    
    @property
    def observation_size(self):
        return self.observation_space.shape[0]

    @property
    def num_tetris_kind(self):
        return 7 if self.align else 19

    @property
    def action_size(self):
        return self.action_space.n
    
    @property
    def game_board_width(self):
        return 10

    @property
    def game_board_height(self):
        return 20

class Tetris:
    piece_colors = [
        (0, 0, 0),
        (255, 255, 0),
        (147, 88, 254),
        (54, 175, 144),
        (255, 0, 0),
        (102, 217, 238),
        (254, 151, 32),
        (0, 0, 255),
        (255, 255, 0),
        (255, 255, 0),
        (255, 255, 0),
        (147, 88, 254),
        (54, 175, 144),
        (255, 0, 0),
        (102, 217, 238),
        (102, 217, 238),
        (102, 217, 238),
        (254, 151, 32),
        (254, 151, 32),
        (254, 151, 32),
    ]

    pieces = [
        [[1, 1],
         [1, 1]],

        [[0, 2, 0],
         [2, 2, 2]],

        [[0, 3, 3],
         [3, 3, 0]],

        [[4, 4, 0],
         [0, 4, 4]],

        [[5],
        [5], 
        [5],
        [5]],

        [[0, 0, 6],
         [6, 6, 6]],

        [[7, 0, 0],
         [7, 7, 7]],

        [[8, 8, 8],
         [0, 8, 0]],

        [[0, 9],
         [9, 9],
         [0, 9]],

        [[10, 0],
         [10, 10],
         [10, 0]],

        [[11, 0 ],
         [11, 11],
         [0 , 11]],

        [[0 , 12],
         [12, 12],
         [12, 0 ]],

        [[13, 13, 13, 13]],

        [[14, 14, 14],
         [14, 0, 0]],

        [[15, 0],
         [15, 0],
         [15, 15]],

        [[16, 16],
         [0,  16],
         [0,  16]],

        [[17, 17, 17],
         [0,  0,  17]],

        [[18, 18],
         [18, 0],
         [18, 0]],

        [[0,  19],
         [0,  19],
         [19, 19]],

    ]

    def __init__(self, height=20, width=10, block_size=20, 
                simplified_feature=False, sim_rom_mode=False, shuffle=True):
        self.height = height
        self.width = width
        self.block_size = block_size
        self.extra_board = np.ones((self.height * self.block_size, self.width * int(self.block_size / 2), 3),
                                   dtype=np.uint8) * np.array([204, 204, 255], dtype=np.uint8)
        self.text_color = (200, 20, 220)
        self.simplified_feature = simplified_feature
        self.sim_rom_mode = sim_rom_mode
        self.shuffle = shuffle
        self.reset()

    def reset(self):
        self.board = [[0] * self.width for _ in range(self.height)]
        self.score = 0
        self.tetrominoes = 0
        self.cleared_lines = 0
        self.bag = list(range(len(self.pieces)))
        if self.shuffle:
            random.shuffle(self.bag)
        self.ind = self.bag.pop()
        self.piece = [row[:] for row in self.pieces[self.ind]]
        self.current_pos = {"x": self.width // 2 - len(self.piece[0]) // 2, "y": 0}
        self.gameover = False
        return self.get_state_properties(self.board)

    def rotate(self, piece):
        num_rows_orig = num_cols_new = len(piece)
        num_rows_new = len(piece[0])
        rotated_array = []

        for i in range(num_rows_new):
            new_row = [0] * num_cols_new
            for j in range(num_cols_new):
                new_row[j] = piece[(num_rows_orig - 1) - j][i]
            rotated_array.append(new_row)
        return rotated_array

    def get_state_properties(self, board):
        lines_cleared, board = self.check_cleared_rows(board)
        if self.simplified_feature:
            holes = self.get_holes(board)
            bumpiness, height = self.get_bumpiness_and_height(board)

            return torch.FloatTensor([lines_cleared, holes, bumpiness, height])
        else:
            num_holes = self.get_holes(board)
            bumpiness, height = self.get_bumpiness_and_height(board)
            heights = self.get_height_vector(board)
            holes = self.get_hole_vector(board, heights)
            return torch.FloatTensor(np.concatenate((heights, holes, 
                                [num_holes, bumpiness, height, lines_cleared, self.ind])))
            # return [torch.FloatTensor(np.concatenate((heights, holes, [lines_cleared]))),board,np.sum(np.minimum(board,1), axis=0)]

    def get_height_vector(self, board):
        board = np.array(board)
        mask = board != 0
        invert_heights = np.where(mask.any(axis=0), np.argmax(mask, axis=0), self.height)
        heights = self.height - invert_heights
        return heights
    
    def get_hole_vector(self, board, heights):
        board = np.array(board)
        total = np.sum(np.minimum(board,1), axis=0)
        return heights - total

    def get_holes(self, board):
        num_holes = 0
        for col in zip(*board):
            row = 0
            while row < self.height and col[row] == 0:
                row += 1
            num_holes += len([x for x in col[row + 1:] if x == 0])
        return num_holes

    def get_bumpiness_and_height(self, board):
        board = np.array(board)
        mask = board != 0
        invert_heights = np.where(mask.any(axis=0), np.argmax(mask, axis=0), self.height)
        heights = self.height - invert_heights
        total_height = np.sum(heights)
        currs = heights[:-1]
        nexts = heights[1:]
        diffs = np.abs(currs - nexts)
        total_bumpiness = np.sum(diffs)
        return total_bumpiness, total_height

    def get_next_states(self):
        states = {}
        piece_id = self.ind
        curr_piece = [row[:] for row in self.piece]
        if piece_id == 0:  # O piece
            num_rotations = 1
        elif piece_id == 2 or piece_id == 3 or piece_id == 4:
            num_rotations = 2
        elif piece_id == 10 or piece_id == 11 or piece_id == 12:
            num_rotations = 2
        else:
            num_rotations = 4

        for i in range(num_rotations):
            valid_xs = self.width - len(curr_piece[0])
            for x in range(valid_xs + 1):
                piece = [row[:] for row in curr_piece]
                if self.sim_rom_mode:
                    x = self.sim_rom_x(x, self.current_pos['x'], piece)
                pos = {"x": x, "y": 0}
                while not self.check_collision(piece, pos):
                    pos["y"] += 1
                self.truncate(piece, pos)
                board = self.store(piece, pos)
                states[(x, i)] = self.get_state_properties(board)
            curr_piece = self.rotate(curr_piece)
        return states

    def sim_rom_x(self, x, current_x, piece):
        # collision detection
        new_x = x
        if x < current_x:
            offset = current_x - x
            for offset_i in range(1, offset+1):
                tmp = current_x - offset_i
                pos = {"x": tmp, "y": offset_i}
                if self.check_collision(piece, pos):
                    new_x = tmp + 1
                    break
        elif x > current_x:
            offset = x - current_x
            for offset_i in range(1, offset+1):
                tmp = current_x + offset_i
                pos = {"x": tmp, "y": offset_i}
                if self.check_collision(piece, pos):
                    new_x = tmp - 1
                    break
        pos = {"x": current_x, "y": 0}
        if self.check_collision(piece, pos):
            new_x = current_x
        return new_x

    def _get_board(self):
        board = self.get_current_board_state()
        board = (np.array(board) > 0).astype(int)
        return board

    def _get_board_ram(self):
        board = [x[:] for x in self.board]
        board = (np.array(board) > 0).astype(int)
        return board

    def update_board(self, board : np.ndarray):
        # Warning: will remove all the piece indices inside the board (set to 1)
        self.board = board.tolist()

    def get_current_board_state(self):
        board = [x[:] for x in self.board]
        for y in range(len(self.piece)):
            for x in range(len(self.piece[y])):
                board[y + self.current_pos["y"]][x + self.current_pos["x"]] = self.piece[y][x]
        return board

    def new_piece(self):
        if not len(self.bag):
            self.bag = list(range(len(self.pieces)))
            if self.shuffle:
                random.shuffle(self.bag)
        self.ind = self.bag.pop()
        self.piece = [row[:] for row in self.pieces[self.ind]]
        self.current_pos = {"x": self.width // 2 - len(self.piece[0]) // 2,
                            "y": 0
                            }
        if self.check_collision(self.piece, self.current_pos):
            self.gameover = True

    def set_new_piece(self, ind):
        self.ind = ind
        self.piece = [row[:] for row in self.pieces[self.ind]]
        self.current_pos = {"x": self.width // 2 - len(self.piece[0]) // 2,
                            "y": 0
                            }
        if self.check_collision(self.piece, self.current_pos):
            self.gameover = True

    def check_collision(self, piece, pos):
        future_y = pos["y"] + 1
        for y in range(len(piece)):
            for x in range(len(piece[y])):
                if future_y + y > self.height - 1 or self.board[future_y + y][pos["x"] + x] and piece[y][x]:
                    return True
        return False

    def truncate(self, piece, pos):
        gameover = False
        last_collision_row = -1
        for y in range(len(piece)):
            for x in range(len(piece[y])):
                if self.board[pos["y"] + y][pos["x"] + x] and piece[y][x]:
                    if y > last_collision_row:
                        last_collision_row = y

        if pos["y"] - (len(piece) - last_collision_row) < 0 and last_collision_row > -1:
            while last_collision_row >= 0 and len(piece) > 1:
                gameover = True
                last_collision_row = -1
                del piece[0]
                for y in range(len(piece)):
                    for x in range(len(piece[y])):
                        if self.board[pos["y"] + y][pos["x"] + x] and piece[y][x] and y > last_collision_row:
                            last_collision_row = y
        return gameover

    def store(self, piece, pos):
        board = [x[:] for x in self.board]
        for y in range(len(piece)):
            for x in range(len(piece[y])):
                if piece[y][x] and not board[y + pos["y"]][x + pos["x"]]:
                    board[y + pos["y"]][x + pos["x"]] = piece[y][x]
        return board

    def check_cleared_rows(self, board):
        to_delete = []
        for i, row in enumerate(board[::-1]):
            if 0 not in row:
                to_delete.append(len(board) - 1 - i)
        if len(to_delete) > 0:
            board = self.remove_row(board, to_delete)
        return len(to_delete), board

    def remove_row(self, board, indices):
        for i in indices[::-1]:
            del board[i]
            board = [[0 for _ in range(self.width)]] + board
        return board

    def step(self, action, render=True, video=None):
        x, num_rotations = action
        self.current_pos = {"x": x, "y": 0}
        for _ in range(num_rotations):
            self.piece = self.rotate(self.piece)

        while not self.check_collision(self.piece, self.current_pos):
            self.current_pos["y"] += 1
            self.render_write(render, video)

        overflow = self.truncate(self.piece, self.current_pos)
        if overflow:
            self.gameover = True

        self.board = self.store(self.piece, self.current_pos)

        lines_cleared, self.board = self.check_cleared_rows(self.board)
        score = 1 + (lines_cleared ** 2) * self.width
        self.score += score
        self.tetrominoes += 1
        self.cleared_lines += lines_cleared
        if not self.gameover:
            self.new_piece()
        if self.gameover:
            # score = -2
            self.score -= 2
        reward = lines_cleared * 0.25
        return reward, self.gameover

    def render_write(self, render=True, video=None):
        if not self.gameover:
            img = [self.piece_colors[p] for row in self.get_current_board_state() for p in row]
        else:
            img = [self.piece_colors[p] for row in self.board for p in row]
        img = np.array(img).reshape((self.height, self.width, 3)).astype(np.uint8)
        img = img[..., ::-1]
        img = Image.fromarray(img, "RGB")

        # fix antialiasing
        img = img.resize((self.width * self.block_size, self.height * self.block_size), 0)
        img = np.array(img)
        img[[i * self.block_size for i in range(self.height)], :, :] = 0
        img[:, [i * self.block_size for i in range(self.width)], :] = 0

        img = np.concatenate((img, self.extra_board), axis=1)


        cv2.putText(img, "Score:", (self.width * self.block_size + int(self.block_size / 2), self.block_size),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=self.text_color)
        cv2.putText(img, str(self.score),
                    (self.width * self.block_size + int(self.block_size / 2), 2 * self.block_size),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=self.text_color)

        cv2.putText(img, "Pieces:", (self.width * self.block_size + int(self.block_size / 2), 4 * self.block_size),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=self.text_color)
        cv2.putText(img, str(self.tetrominoes),
                    (self.width * self.block_size + int(self.block_size / 2), 5 * self.block_size),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=self.text_color)

        cv2.putText(img, "Lines:", (self.width * self.block_size + int(self.block_size / 2), 7 * self.block_size),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=self.text_color)
        cv2.putText(img, str(self.cleared_lines),
                    (self.width * self.block_size + int(self.block_size / 2), 8 * self.block_size),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=self.text_color)

        if video:
            video.write(img)
            
        if render:
            cv2.imshow("Deep Q-Learning Tetris", img)
            cv2.waitKey(1)


if __name__ == "__main__":
    env = gym.make('Tetris-v0')
    env = SymbolTetrisSimple(env)
    obs = env.reset()
    obs, reward, done = env.step(0)
    