# Learn to Play Tetris with Deep RL

## Prepare

```
# prepare env
bash ./prepare.sh
```

## How-to
#### dqn
```
cd dqn
python train.py
# Test dqn on Simplified Env 
python test.py --ckpt_name gamma0999/tetris_6000.pth
# Test dqn on Pygame Env
python sim2rom.py --ckpt_name gamma0999/tetris_6000.pth
```

#### sac
```
cd sac
python train.py
```
#### ppo
```
cd ppo
python train.py
```

#### dreamer
```
cd dreamer
python main.py --algo dreamer --collect_interval 200 --test_interval 5 --add_reward True --small_image True --expl_amount 0.3 --binary_image True
```
#### curl
```
cd curl
python main.py --inc_level --reward_mode 0
```
#### drq
```
cd drq
python train.py
```
#### plan2explore
```
cd plan2explore
python simple.py --collect-interval 100 --test-interval 5  --env Tetris-v0 --symbolic-env  --expl_amount 0.3 --algo p2e
```
#### lucid-dreamer
```
cd plan2explore
python simple.py --collect-interval 100 --test-interval 5  --env Tetris-v0 --symbolic-env  --expl_amount 0.1 --algo dreamer --experience-buffer buffer.npz --use-reward --clone
```

## Acknowledgment

- Dreamer implementation is based on https://github.com/danijar/dreamer and https://github.com/yusukeurakami/dreamer-pytorch 
- DQN implementation is based on https://github.com/uvipen/Tetris-deep-Q-learning-pytorch
- DrQ is based on https://github.com/denisyarats/drq
- CURL-Rainbow is based on https://github.com/aravindsrinivas/curl_rainbow
- SAC is based on https://github.com/ku2482/sac-discrete.pytorch
- PPO is based on https://github.com/cuhkrlcourse/ierg5350-assignment 
- Plan2Explore is based on https://github.com/yusukeurakami/plan2explore-pytorch and https://github.com/ramanans1/plan2explore 
- gym_tetris is adopted from https://github.com/Kautenja/gym-tetris
- gym_tetris_simple is adopted from https://github.com/lusob/gym-tetris
- tetris.py is based on https://github.com/uvipen/Tetris-deep-Q-learning-pytorch/blob/master/src/tetris.py