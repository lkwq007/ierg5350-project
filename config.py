import argparse
from env import CONTROL_SUITE_ENVS, GYM_ENVS, NES_ENVS
from utils import str2bool
from torch.nn import functional as F

parser = argparse.ArgumentParser(description='PlaNet or Dreamer')
parser.add_argument('--algo', type=str, default='dreamer',
                    help='planet or dreamer')
parser.add_argument('--id', type=str, default='default', help='Experiment ID')
parser.add_argument('--seed', type=int, default=1,
                    metavar='S', help='Random seed')
parser.add_argument('--disable_cuda', action='store_true', help='Disable CUDA')
parser.add_argument('--gpu_no', type=int, default=0, help='GPU id')
parser.add_argument('--env', type=str, default='TetrisA-v0', choices=NES_ENVS +
                    GYM_ENVS + CONTROL_SUITE_ENVS, help='Gym/Control Suite environment')
parser.add_argument('--max_episode_length', type=int,
                    default=1000, metavar='T', help='Max episode length')
# Original implementation has an unlimited buffer size, but 1 million is the max experience collected anyway
parser.add_argument('--experience_size', type=int,
                    default=1000000, metavar='D', help='Experience replay size')
parser.add_argument('--cnn_activation_function', type=str, default='relu',
                    choices=dir(F), help='Model activation function for a convolution layer')
parser.add_argument('--dense_activation_function', type=str, default='elu',
                    choices=dir(F), help='Model activation function a dense layer')
# Note that the default encoder for visual observations outputs a 1024D vector; for other embedding sizes an additional fully-connected layer is used
parser.add_argument('--embedding_size', type=int, default=1024,
                    metavar='E', help='Observation embedding size')
parser.add_argument('--hidden_size', type=int, default=200,
                    metavar='H', help='Hidden size')
parser.add_argument('--belief_size', type=int, default=200,
                    metavar='H', help='Belief/hidden size')
parser.add_argument('--state_size', type=int, default=30,
                    metavar='Z', help='State/latent size')
parser.add_argument('--action_repeat', type=int, default=1,
                    metavar='R', help='Action repeat')
parser.add_argument('--action_dist', type=str,
                    default='onehot', help='Action Distribution')
parser.add_argument('--episodes', type=int, default=1000,
                    metavar='E', help='Total number of episodes')
parser.add_argument('--seed_episodes', type=int, default=5,
                    metavar='S', help='Seed episodes')
parser.add_argument('--collect_interval', type=int,
                    default=100, metavar='C', help='Collect interval')
parser.add_argument('--batch_size', type=int, default=50,
                    metavar='B', help='Batch size')
parser.add_argument('--chunk_size', type=int, default=50,
                    metavar='L', help='Chunk size')
parser.add_argument('--worldmodel_LogProbLoss', type=str2bool, default=True,
                    help='use LogProb loss for observation_model and reward_model training')
parser.add_argument('--global_kl_beta', type=float, default=0,
                    metavar='βg', help='Global KL weight (0 to disable)')
parser.add_argument('--free_nats', type=float, default=3,
                    metavar='F', help='Free nats')
parser.add_argument('--bit_depth', type=int, default=5,
                    metavar='B', help='Image bit depth (quantisation)')
parser.add_argument('--model_learning_rate', type=float,
                    default=1e-3, metavar='α', help='Learning rate')
parser.add_argument('--actor_learning_rate', type=float,
                    default=8e-5, metavar='α', help='Learning rate')
parser.add_argument('--value_learning_rate', type=float,
                    default=8e-5, metavar='α', help='Learning rate')
parser.add_argument('--learning_rate_schedule', type=int, default=0, metavar='αS',
                    help='Linear learning rate schedule (optimisation steps from 0 to final learning rate; 0 to disable)')
parser.add_argument('--adam_epsilon', type=float, default=1e-7,
                    metavar='ε', help='Adam optimizer epsilon value')
# Note that original has a linear learning rate decay, but it seems unlikely that this makes a significant difference
parser.add_argument('--grad_clip_norm', type=float,
                    default=100.0, metavar='C', help='Gradient clipping norm')
parser.add_argument('--planning_horizon', type=int, default=15,
                    metavar='H', help='Planning horizon distance')
parser.add_argument('--discount', type=float, default=0.99,
                    metavar='H', help='Planning horizon distance')
parser.add_argument('--disclam', type=float, default=0.95,
                    metavar='H', help='discount rate to compute return')
parser.add_argument('--optimisation_iters', type=int, default=10,
                    metavar='I', help='Planning optimisation iterations')
parser.add_argument('--expl_type', type=str,
                    default='epsilon_greedy', help='Exploration Decay Init Value')
parser.add_argument('--expl_amount', type=float, default=0.4,
                    help='Exploration Decay Init Value')
parser.add_argument('--expl_decay', type=float, default=100000.0,
                    help='Exploration Decay Weight')
parser.add_argument('--expl_min', type=float, default=0.1,
                    help='Minimum Exploration Decay Value')
parser.add_argument('--candidates', type=int, default=1000,
                    metavar='J', help='Candidate samples per iteration')
parser.add_argument('--top_candidates', type=int, default=100,
                    metavar='K', help='Number of top candidates to fit')
parser.add_argument('--test', action='store_true', help='Test only')
parser.add_argument('--test_interval', type=int, default=25,
                    metavar='I', help='Test interval (episodes)')
parser.add_argument('--test_episodes', type=int, default=10,
                    metavar='E', help='Number of test episodes')
parser.add_argument('--checkpoint_interval', type=int, default=25,
                    metavar='I', help='Checkpoint interval (episodes)')
parser.add_argument('--checkpoint_experience',
                    action='store_true', help='Checkpoint experience replay')
parser.add_argument('--models', type=str, default='',
                    metavar='M', help='Load model checkpoint')
parser.add_argument('--experience_replay', type=str, default='',
                    metavar='ER', help='Load experience replay')
parser.add_argument('--render', action='store_true', help='Render environment')
parser.add_argument('--image_pad', type=int, default=4,
                    metavar='PAD', help='For image aug')
parser.add_argument('--doubleq', action='store_true', help='enable doubleQ')
parser.add_argument('--pcont', type=str2bool, default=True, help='enable pcont')
parser.add_argument('--pcont_scale', type=float, default=10.0, help='enable pcont')
parser.add_argument('--kl_scale', type=float, default=0.1, help='enable pcont')

parser.add_argument('--small_image', type=str2bool, default=False, help='using 96x96 image')
parser.add_argument('--add_reward', type=str2bool, default=False, help='additional reward')
parser.add_argument('--experience_list', type=str, default='', metavar='ELL', help='Load experience replay')

class Args(object):
    def __init__(self, _parser=parser) -> None:
        super().__init__()
        # Hyperparameters
        self._parser = _parser
        args_dict = vars(self._parser.parse_args())
        for item in args_dict:
            setattr(self, item, args_dict[item])

        print('---- Options ----')
        for k, v in args_dict.items():
            print(k + ': ' + str(v))
        print('--------\n')

        if self.doubleq:
            print("DrQ is used in this training")
        