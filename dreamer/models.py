from typing import Optional, List
import torch
# from torch import jit, nn
from torch import nn
from torch.nn import functional as F
import torch.distributions
from torch.distributions import Normal, Categorical, Bernoulli
from torch.distributions.transforms import Transform, TanhTransform
from torch.distributions.transformed_distribution import TransformedDistribution
import numpy as np

from torch import Tensor
# note that nn.Module is deprecated
# https://discuss.pytorch.org/t/whats-the-difference-between-torch-nn-module-and-torch-jit-scriptmodule/64480
# so I removed all nn.Module in this file and planner
# By Yuan

# Wraps the input tuple for a function to process a time x batch x features sequence in batch x features (assumes one output)


def bottle(f, x_tuple):
    x_sizes = tuple(map(lambda x: x.size(), x_tuple))
    y = f(*map(lambda x: x[0].view(x[1][0] * x[1][1], *x[1][2:]),
               zip(x_tuple, x_sizes)))
    y_size = y.size()
    output = y.view(x_sizes[0][0], x_sizes[0][1], *y_size[1:])
    return output


class TransitionModel(nn.Module):
    __constants__ = ['min_std_dev']

    def __init__(self,
                 belief_size,
                 state_size,
                 action_size,
                 hidden_size,
                 embedding_size,
                 activation_function='elu',  # replaced by elu
                 min_std_dev=0.1):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.min_std_dev = min_std_dev
        self.fc_embed_state_action = nn.Linear(state_size + action_size,
                                               belief_size)
        self.rnn = nn.GRUCell(belief_size, belief_size)
        self.fc_embed_belief_prior = nn.Linear(belief_size, hidden_size)
        self.fc_state_prior = nn.Linear(hidden_size, 2 * state_size)
        self.fc_embed_belief_posterior = nn.Linear(
            belief_size + embedding_size, hidden_size)
        self.fc_state_posterior = nn.Linear(hidden_size, 2 * state_size)
        self.modules = [
            self.fc_embed_state_action, self.fc_embed_belief_prior,
            self.fc_state_prior, self.fc_embed_belief_posterior,
            self.fc_state_posterior
        ]

    # Operates over (previous) state, (previous) actions, (previous) belief, (previous) nonterminals (mask), and (current) observations
    # Diagram of expected inputs and outputs for T = 5 (-x- signifying beginning of output belief/state that gets sliced off):
    # t :  0  1  2  3  4  5
    # o :    -X--X--X--X--X-
    # a : -X--X--X--X--X-
    # n : -X--X--X--X--X-
    # pb: -X-
    # ps: -X-
    # b : -x--X--X--X--X--X-
    # s : -x--X--X--X--X--X-
    # @jit.script_method
    def forward(
            self,
            prev_state: torch.Tensor,
            actions: torch.Tensor,
            prev_belief: torch.Tensor,
            observations: Optional[torch.Tensor] = None,
            nonterminals: Optional[torch.Tensor] = None) -> List[torch.Tensor]:
        '''
    Input: init_belief, init_state:  torch.Size([50, 200]) torch.Size([50, 30])
    Output: beliefs, prior_states, prior_means, prior_std_devs, posterior_states, posterior_means, posterior_std_devs
            torch.Size([49, 50, 200]) torch.Size([49, 50, 30]) torch.Size([49, 50, 30]) torch.Size([49, 50, 30]) torch.Size([49, 50, 30]) torch.Size([49, 50, 30]) torch.Size([49, 50, 30])
    '''
        # Create lists for hidden states (cannot use single tensor as buffer because autograd won't work with inplace writes)
        T = actions.size(0) + 1
        beliefs = [torch.empty(0)] * T
        prior_states = [torch.empty(0)] * T
        prior_means = [torch.empty(0)] * T
        prior_std_devs = [torch.empty(0)] * T
        posterior_states = [torch.empty(0)] * T
        posterior_means = [torch.empty(0)] * T
        posterior_std_devs = [torch.empty(0)] * T
        beliefs[0] = prev_belief
        prior_states[0] = prev_state
        posterior_states[0] = prev_state
        # Loop over time sequence
        for t in range(T - 1):
            # Select appropriate previous state
            _state = prior_states[t] if observations is None else posterior_states[t]  
            # Mask if previous transition was terminal
            _state = _state if nonterminals is None else _state * nonterminals[t]
            # Compute belief (deterministic hidden state)
            hidden = self.act_fn(
                self.fc_embed_state_action(
                    torch.cat([_state, actions[t]], dim=1)))
            beliefs[t + 1] = self.rnn(hidden, beliefs[t])
            # Compute state prior by applying transition dynamics
            hidden = self.act_fn(self.fc_embed_belief_prior(beliefs[t + 1]))
            prior_means[t + 1], _prior_std_dev = torch.chunk(
                self.fc_state_prior(hidden), 2, dim=1)
            prior_std_devs[t +
                           1] = F.softplus(_prior_std_dev) + self.min_std_dev
            prior_states[t + 1] = prior_means[t + 1] + prior_std_devs[
                t + 1] * torch.randn_like(prior_means[t + 1])
            if observations is not None:
                # Compute state posterior by applying transition dynamics and using current observation
                t_ = t - 1  # Use t_ to deal with different time indexing for observations
                hidden = self.act_fn(
                    self.fc_embed_belief_posterior(
                        torch.cat([beliefs[t + 1], observations[t_ + 1]],
                                  dim=1)))
                posterior_means[t + 1], _posterior_std_dev = torch.chunk(
                    self.fc_state_posterior(hidden), 2, dim=1)
                posterior_std_devs[
                    t + 1] = F.softplus(_posterior_std_dev) + self.min_std_dev
                posterior_states[t + 1] = posterior_means[
                    t + 1] + posterior_std_devs[t + 1] * torch.randn_like(
                        posterior_means[t + 1])
        # Return new hidden states
        hidden = [
            torch.stack(beliefs[1:], dim=0),
            torch.stack(prior_states[1:], dim=0),
            torch.stack(prior_means[1:], dim=0),
            torch.stack(prior_std_devs[1:], dim=0)
        ]
        if observations is not None:
            hidden += [
                torch.stack(posterior_states[1:], dim=0),
                torch.stack(posterior_means[1:], dim=0),
                torch.stack(posterior_std_devs[1:], dim=0)
            ]
        return hidden


class VisualObservationModel(nn.Module):
    __constants__ = ['embedding_size']

    def __init__(self,
                 belief_size,
                 state_size,
                 embedding_size,
                 activation_function='relu',
                 small_image=False,binary=False):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.embedding_size = embedding_size
        self.fc1 = nn.Linear(belief_size + state_size, embedding_size)
        self.small_image=small_image
        if small_image:
            self.conv1 = nn.ConvTranspose2d(embedding_size, 128, 5, stride=2)
            self.conv2 = nn.ConvTranspose2d(128, 64, 5, stride=2)
            self.conv3 = nn.ConvTranspose2d(64, 32, 7, stride=2)
            self.conv4 = nn.ConvTranspose2d(32, 1 if binary else 3, 6, stride=3)
        else:
            self.conv1 = nn.ConvTranspose2d(embedding_size, 256, 5, stride=2)
            self.conv2 = nn.ConvTranspose2d(256, 128, 5, stride=2)
            self.conv3 = nn.ConvTranspose2d(128, 64, 5, stride=2)
            self.conv4 = nn.ConvTranspose2d(64, 32, 6, stride=2)
            self.conv5 = nn.ConvTranspose2d(32, 1 if binary else 3, 6, stride=2)
        self.modules = [
            self.fc1, self.conv1, self.conv2, self.conv3, self.conv4
        ]

    # @jit.script_method
    def forward(self, belief, state):
        hidden = self.fc1(torch.cat([belief, state],
                                    dim=1))  # No nonlinearity here
        hidden = hidden.view(-1, self.embedding_size, 1, 1)
        hidden = self.act_fn(self.conv1(hidden))
        hidden = self.act_fn(self.conv2(hidden))
        hidden = self.act_fn(self.conv3(hidden))
        if self.small_image:
            observation = self.conv4(hidden)
        else:
            hidden = self.act_fn(self.conv4(hidden))
            observation = self.conv5(hidden)
        return observation


def ObservationModel(observation_size,
                     belief_size,
                     state_size,
                     embedding_size,
                     activation_function='relu', small_image=False, binary=False):
    return VisualObservationModel(belief_size, state_size, embedding_size,
                                  activation_function, small_image, binary)


class RewardModel(nn.Module):
    def __init__(self,
                 belief_size,
                 state_size,
                 hidden_size,
                 activation_function='relu'):
        # [--belief-size: 200, --hidden-size: 200, --state-size: 30]
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.fc1 = nn.Linear(belief_size + state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.modules = [self.fc1, self.fc2, self.fc3]

    # @jit.script_method
    def forward(self, belief, state):
        x = torch.cat([belief, state], dim=1)
        hidden = self.act_fn(self.fc1(x))
        hidden = self.act_fn(self.fc2(hidden))
        reward = self.fc3(hidden).squeeze(dim=1)
        return reward


class PcontModel(nn.Module):
    def __init__(self,
                 belief_size,
                 state_size,
                 hidden_size,
                 activation_function='relu'):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.fc1 = nn.Linear(belief_size + state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, 1)
        self.modules = [self.fc1, self.fc2, self.fc3, self.fc4]

    # @jit.script_method
    def forward(self, belief, state):
        x = torch.cat([belief, state], dim=1)
        hidden = self.act_fn(self.fc1(x))
        hidden = self.act_fn(self.fc2(hidden))
        hidden = self.act_fn(self.fc3(hidden))
        pcont = self.fc4(hidden).squeeze(dim=1)
        return pcont


def ValueModel(belief_size, state_size, hidden_size, activation_function='relu', doubleQ=False):
    if doubleQ:
        return ValueModelDoubleQ(belief_size, state_size, hidden_size, activation_function)
    else:
        return ValueModelOriginal(belief_size, state_size, hidden_size, activation_function)


class ValueModelOriginal(nn.Module):
    def __init__(self,
                 belief_size,
                 state_size,
                 hidden_size,
                 activation_function='relu'):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.fc1 = nn.Linear(belief_size + state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, 1)
        self.modules = [self.fc1, self.fc2, self.fc3, self.fc4]

    # @jit.script_method
    def forward(self, belief, state):
        x = torch.cat([belief, state], dim=1)
        hidden = self.act_fn(self.fc1(x))
        hidden = self.act_fn(self.fc2(hidden))
        hidden = self.act_fn(self.fc3(hidden))
        reward = self.fc4(hidden).squeeze(dim=1)
        return reward


class ValueModelDoubleQ(nn.Module):
    # note that double Q need exta modification in main.py
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.Q1 = ValueModelOriginal(*args, **kwargs)
        self.Q2 = ValueModelOriginal(*args, **kwargs)
        self.modules = self.Q1.modules+self.Q2.modules
        # self.modules = [self.fc1, self.fc2, self.fc3, self.fc4]

    # @jit.script_method
    def forward(self, belief, state):
        q1 = self.Q1(belief, state)
        q2 = self.Q2(belief, state)
        return q1, q2


class ActorModel(nn.Module):
    def __init__(self,
                 belief_size,
                 state_size,
                 hidden_size,
                 action_size,
                 dist='tanh_normal',
                 activation_function='elu',
                 min_std=1e-4,
                 init_std=5.0,
                 mean_scale=5.0):
        super().__init__()
        self._dist = dist
        self._min_std = min_std
        self._init_std = init_std
        self._mean_scale = mean_scale

        self.act_fn = getattr(F, activation_function)
        self.fc1 = nn.Linear(belief_size + state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        if self._dist == 'tanh_normal':
            self.fc5 = nn.Linear(hidden_size, 2 * action_size)
        elif self._dist == 'onehot':
            self.fc5 = nn.Linear(hidden_size, action_size)
        else:
            raise NotImplementedError(self._dist)

        self.modules = [self.fc1, self.fc2, self.fc3, self.fc4, self.fc5]

    # @jit.script_method
    def forward(self, belief, state):

        x = torch.cat([belief, state], dim=1)
        hidden = self.act_fn(self.fc1(x))
        hidden = self.act_fn(self.fc2(hidden))
        hidden = self.act_fn(self.fc3(hidden))
        hidden = self.act_fn(self.fc4(hidden))
        actor_out = self.fc5(hidden).squeeze(dim=1)

        return actor_out

    def get_action(self, belief, state, det=False):
        actor_out = self.forward(belief, state)
        if self._dist == 'tanh_normal':
            # actor_out.size() == (N x (action_size * 2))
            # replace the below workaround
            raw_init_std = np.log(np.exp(self._init_std) - 1)
            # tmp = torch.tensor(self._init_std,
            #                    device=actor_out.get_device())
            # raw_init_std = torch.log(torch.exp(tmp) - 1)
            action_mean, action_std_dev = torch.chunk(actor_out, 2, dim=1)
            action_mean = self._mean_scale * torch.tanh(
                action_mean / self._mean_scale)
            action_std = F.softplus(action_std_dev +
                                    raw_init_std) + self._min_std

            dist = Normal(action_mean, action_std)
            dist = TransformedDistribution(dist, TanhBijector())
            dist = torch.distributions.Independent(dist, 1)
            dist = SampleDist(dist)
        elif self._dist == 'onehot':
            # actor_out.size() == (N x action_size)
            # fix for RuntimeError: CUDA error: device-side assert triggered
            actor_out=(torch.tanh(actor_out)+1.0) * 0.5
            dist = Categorical(logits=actor_out)
            dist = OneHotDist(dist)
        else:
            raise NotImplementedError(self._dist)
        if det:
            return dist.mode()
        else:
            return dist.sample()


class VisualEncoder(nn.Module):
    __constants__ = ['embedding_size']

    def __init__(self, embedding_size, activation_function='relu',small_image=False, binary=False):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        # perhaps we need larger embedding_size
        self.embedding_size = embedding_size
        # kernel 4 seems strange
        if small_image:
            # 96x96
            self.conv1 = nn.Conv2d(1 if binary else 3, 32, 4, stride=3)
            self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
            self.conv3 = nn.Conv2d(64, 128, 4, stride=2)
            self.conv4 = nn.Conv2d(128, 256, 4, stride=2)
        else:
            self.conv1 = nn.Conv2d(1 if binary else 3, 32, 5, stride=2)
            self.conv2 = nn.Conv2d(32, 64, 5, stride=3)
            self.conv3 = nn.Conv2d(64, 128, 5, stride=2)
            self.conv4 = nn.Conv2d(128, 256, 5, stride=2)
        self.fc = nn.Identity() if embedding_size == 1024 else nn.Linear(
            1024, embedding_size)
        self.modules = [self.conv1, self.conv2, self.conv3, self.conv4]

    # @jit.script_method
    def forward(self, observation):
        hidden = self.act_fn(self.conv1(observation))
        hidden = self.act_fn(self.conv2(hidden))
        hidden = self.act_fn(self.conv3(hidden))
        hidden = self.act_fn(self.conv4(hidden))
        hidden = hidden.view(-1, 1024)
        hidden = self.fc(
            hidden
        )  # Identity if embedding size is 1024 else linear projection
        return hidden


def Encoder(observation_size, embedding_size, activation_function='relu', small_image=False, binary=False):
    return VisualEncoder(embedding_size, activation_function, small_image, binary)


# "atanh", "TanhBijector" and "SampleDist" are from the following repo
# https://github.com/juliusfrost/dreamer-pytorch
def atanh(x):
    return 0.5 * torch.log((1 + x) / (1 - x))


class TanhBijector(torch.distributions.Transform):
    def __init__(self):
        super().__init__()
        self.bijective = True

    @property
    def sign(self):
        return 1.

    def _call(self, x):
        return torch.tanh(x)

    def _inverse(self, y: torch.Tensor):
        y = torch.where((torch.abs(y) <= 1.),
                        torch.clamp(y, -0.99999997, 0.99999997), y)
        y = atanh(y)
        return y

    def log_abs_det_jacobian(self, x, y):
        return 2. * (np.log(2) - x - F.softplus(-2. * x))


class SampleDist:
    def __init__(self, dist, samples=100):
        self._dist = dist
        self._samples = samples

    @property
    def name(self):
        return 'SampleDist'

    def __getattr__(self, name):
        return getattr(self._dist, name)

    def mode(self):
        dist = self._dist.expand((self._samples, *self._dist.batch_shape))
        sample = dist.rsample()
        logprob = dist.log_prob(sample)
        batch_size = sample.size(1)
        feature_size = sample.size(2)
        indices = torch.argmax(logprob,
                               dim=0).reshape(1, batch_size, 1).expand(
                                   1, batch_size, feature_size)
        return torch.gather(sample, 0, indices).squeeze(0)

    def entropy(self):
        dist = self._dist.expand((self._samples, *self._dist.batch_shape))
        sample = dist.rsample()
        logprob = dist.log_prob(sample)
        return -torch.mean(logprob, 0)

    def sample(self):
        return self.rsample()


class OneHotDist:
    def __init__(self, dist):
        self._dist = dist
        self._num_classes = dist.probs.shape[-1]

    @property
    def name(self):
        return 'OneHotDist'

    def __getattr__(self, name):
        return getattr(self._dist, name)

    def mode(self):
        return self._one_hot(self._dist.probs.argmax(dim=1))

    def sample(self):
        indices = self._dist.sample()
        sample = self._one_hot(indices)
        probs = self._dist.probs
        sample += (probs - probs.detach()).float()  # make probs differentiable
        return sample

    def _one_hot(self, indices):
        return F.one_hot(indices, self._num_classes).float()
