from tetris import Tetris
import random
import cv2

env = Tetris(width=10,height=20,block_size=30)
state = env.reset()
print(state)
for i in range(10):
    next_steps = env.get_next_states()
    next_actions, next_states = zip(*next_steps.items())
    idx = random.randrange(0, len(next_steps))
    if isinstance(next_states[idx],list):
        [print(item) for item in next_states[idx]]
    else:
        print(next_states[idx])
    reward, done = env.step(next_actions[idx], render=True)
    cv2.waitKey(0)