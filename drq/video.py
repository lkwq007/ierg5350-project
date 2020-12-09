import os
import sys

import imageio
import numpy as np

import utils


class VideoRecorder(object):
    def __init__(self, root_dir, height=256, width=256, fps=10):
        self.save_dir = utils.make_dir(root_dir, 'video') if root_dir else None
        self.height = height
        self.width = width
        self.fps = fps
        self.frames = []

    def init(self, enabled=True):
        self.frames = []
        self.enabled = self.save_dir is not None and enabled

    def record(self, obs):
        if self.enabled:
            # removed incompatible kwargs for tetris
            # frame = env.render(mode='rgb_array')
            self.frames.append(obs.transpose(1,2,0)[:,:,0:3])

    def save(self, file_name):
        if self.enabled:
            path = os.path.join(self.save_dir, file_name)
            imageio.mimsave(path, self.frames, fps=self.fps)
