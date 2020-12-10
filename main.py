from config import Args
import os
import numpy as np
import torch
from torch import nn, optim
from torch.distributions import Normal, Categorical, Bernoulli
from torch.distributions.kl import kl_divergence
from torch.nn import functional as F
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
from env import CONTROL_SUITE_ENVS, Env, GYM_ENVS, EnvBatcher, NES_ENVS
from memory import ExperienceReplay
from models import bottle, Encoder, ObservationModel, RewardModel, PcontModel, TransitionModel, ValueModel, ActorModel
from planner import MPCPlanner
from utils import *
from torch.utils.tensorboard import SummaryWriter
from env_utils import make_envs
# Setup
args = Args()
setup_my_seed(args)
device = get_my_device(args)

# Recorder
results_dir = os.path.join('results', '{}_{}'.format(args.env, args.id))
os.makedirs(results_dir, exist_ok=True)
writer = SummaryWriter(results_dir + "/{}_{}_log".format(args.env, args.id))
metrics = {
    'steps': [], 'episodes': [], 'train_rewards': [], 'test_episodes': [], 'test_rewards': [], 'observation_loss': [], 'reward_loss': [], 'kl_loss': [], 'actor_loss': [], 'value_loss': []
}

# Initialise training environment and experience replay memory
env = Env(args.env, args.seed, args.max_episode_length, args.action_repeat,
          args.bit_depth)
test_envs = EnvBatcher(Env, (args.env, args.seed, args.max_episode_length,
                                args.action_repeat, args.bit_depth), {},
                               args.test_episodes)
# test_envs2 = make_envs()
if args.experience_replay != '' and os.path.exists(args.experience_replay):
    D = torch.load(args.experience_replay)
    metrics['steps'], metrics['episodes'] = [D.steps] * D.episodes, list(
        range(1, D.episodes + 1))
elif not args.test:
    D = ExperienceReplay(args.experience_size, env.observation_size,
                         env.action_size, args.bit_depth, device)
    # Initialise dataset D with S random seed episodes
    for s in range(1, args.seed_episodes + 1):
        observation, done, t = env.reset(), False, 0
        while not done:
            action = env.sample_random_action()
            next_observation, reward, done = env.step(action)
            D.append(observation, action, reward, done)
            observation = next_observation
            t += 1
        metrics['steps'].append(t * args.action_repeat + (
            0 if len(metrics['steps']) == 0 else metrics['steps'][-1]))
        metrics['episodes'].append(s)

# Initialise model parameters randomly
transition_model = TransitionModel(
    args.belief_size, args.state_size, env.action_size, args.hidden_size,
    args.embedding_size, args.dense_activation_function).to(device)
observation_model = ObservationModel(
    env.observation_size, args.belief_size, args.state_size,
    args.embedding_size, args.cnn_activation_function).to(device)
reward_model = RewardModel(
    args.belief_size, args.state_size, args.hidden_size,
    args.dense_activation_function).to(device)
pcont_model = PcontModel(
    args.belief_size, args.state_size, args.hidden_size, args.dense_activation_function).to(device)
encoder = Encoder(env.observation_size, args.embedding_size,
                  args.cnn_activation_function).to(device)
actor_model = ActorModel(args.belief_size, args.state_size, args.hidden_size,
                         env.action_size, args.action_dist,
                         args.dense_activation_function).to(device)
value_model = ValueModel(args.belief_size, args.state_size, args.hidden_size,
                         args.dense_activation_function).to(device)
# Param List
param_list = list(transition_model.parameters()) + list(
    observation_model.parameters()) + list(reward_model.parameters()) + list(
        encoder.parameters())
if args.pcont:
    param_list += list(pcont_model.parameters())
model_optimizer = optim.Adam(
    param_list,
    lr=0 if args.learning_rate_schedule != 0 else args.model_learning_rate,
    eps=args.adam_epsilon)
actor_optimizer = optim.Adam(
    actor_model.parameters(),
    lr=0 if args.learning_rate_schedule != 0 else args.actor_learning_rate,
    eps=args.adam_epsilon)
value_optimizer = optim.Adam(
    value_model.parameters(),
    lr=0 if args.learning_rate_schedule != 0 else args.value_learning_rate,
    eps=args.adam_epsilon)
if args.models != '' and os.path.exists(args.models):
    model_dicts = torch.load(args.models)
    transition_model.load_state_dict(model_dicts['transition_model'])
    observation_model.load_state_dict(model_dicts['observation_model'])
    reward_model.load_state_dict(model_dicts['reward_model'])
    if args.pcont:
        pcont_model.load_state_dict(model_dicts['pcont_model'])
    encoder.load_state_dict(model_dicts['encoder'])
    actor_model.load_state_dict(model_dicts['actor_model'])
    value_model.load_state_dict(model_dicts['value_model'])
    model_optimizer.load_state_dict(model_dicts['model_optimizer'])
    actor_optimizer.load_state_dict(model_dicts['actor_optimizer'])
    value_optimizer.load_state_dict(model_dicts['value_optimizer'])

if args.algo == "dreamer":
    print("DREAMER")
    planner = actor_model
else:
    planner = MPCPlanner(env.action_size, args.planning_horizon,
                         args.optimisation_iters, args.candidates,
                         args.top_candidates, transition_model, reward_model)
global_prior = Normal(
    torch.zeros(args.batch_size, args.state_size, device=device),
    torch.ones(args.batch_size, args.state_size,
               device=device))  # Global prior N(0, I)
free_nats = torch.full(
    (1, ), args.free_nats,
    device=device)  # Allowed deviation in KL divergence


def update_belief_and_act(args,
                          env,
                          planner,
                          transition_model,
                          encoder,
                          belief,
                          posterior_state,
                          action,
                          observation,
                          explore=False,
                          step=0):
    # Infer belief over current state q(s_t|o≤t,a<t) from the history
    belief, _, _, _, posterior_state, _, _ = transition_model(
        posterior_state, action.unsqueeze(dim=0), belief,
        encoder(observation).unsqueeze(
            dim=0))  # Action and observation need extra time dimension
    belief, posterior_state = belief.squeeze(dim=0), posterior_state.squeeze(
        dim=0)  # Remove time dimension from belief/state
    if args.algo == "dreamer":
        action = planner.get_action(belief, posterior_state, det=not (explore))
    else:
        action = planner(
            belief,
            posterior_state)  # Get action from planner(q(s_t|o≤t,a<t), p)
    if explore:
        action = exploration(args, action, step)
    next_observation, reward, done = env.step(
        action.cpu() if isinstance(env, EnvBatcher) else action[0].cpu(
        ))  # Perform environment step (action repeats handled internally)
    return belief, posterior_state, action, next_observation, reward, done


def exploration(args, action, step):
    amount = args.expl_amount
    if args.expl_decay:
        amount *= 0.5**(float(step) / args.expl_decay)
    if args.expl_min:
        amount = max(args.expl_min, amount)
    if args.expl_type == 'additive_gaussian':
        action = torch.clamp(Normal(action, amount).rsample(), -1, 1)
    elif args.expl_type == 'epsilon_greedy':
        prob = torch.ones_like(action)
        prob = prob / torch.sum(prob)
        indices = Categorical(prob).sample()
        action = torch.where(
            torch.rand(action.shape[:1], device=action.get_device()) < amount,
            F.one_hot(indices, action.shape[-1]).float(), action)
    else:
        raise NotImplementedError(args.expl_type)
    return action


# Testing only
if args.test:
    # Set models to eval mode
    transition_model.eval()
    reward_model.eval()
    encoder.eval()
    with torch.no_grad():
        total_reward = 0
        for _ in tqdm(range(args.test_episodes)):
            observation = env.reset()
            belief, posterior_state, action = torch.zeros(
                1, args.belief_size, device=device), torch.zeros(
                    1, args.state_size,
                    device=device), torch.zeros(1,
                                                     env.action_size,
                                                     device=device)
            pbar = tqdm(range(args.max_episode_length // args.action_repeat))
            for t in pbar:
                belief, posterior_state, action, observation, reward, done = update_belief_and_act(
                    args, env, planner, transition_model, encoder,
                    belief, posterior_state, action,
                    observation.to(device))
                total_reward += reward
                if args.render:
                    env.render()
                if done:
                    pbar.close()
                    break
    print('Average Reward:', total_reward / args.test_episodes)
    env.close()
    quit()

# Training (and testing)
for episode in tqdm(range(metrics['episodes'][-1] + 1, args.episodes + 1),
                    total=args.episodes,
                    initial=metrics['episodes'][-1] + 1):
    # Model fitting
    losses = []
    model_modules = transition_model.modules + encoder.modules + observation_model.modules + reward_model.modules
    if args.pcont:
        model_modules += pcont_model.modules
    print("training loop")
    for s in tqdm(range(args.collect_interval)):
        # Draw sequence chunks {(o_t, a_t, r_t+1, terminal_t+1)} ~ D uniformly at random from the dataset (including terminal flags)
        observations, actions, rewards, nonterminals = D.sample(
            args.batch_size,
            args.chunk_size)  # Transitions start at time t = 0
        # Create initial belief and state for time t = 0
        init_belief, init_state = torch.zeros(args.batch_size,
                                              args.belief_size,
                                              device=device), torch.zeros(
                                                  args.batch_size,
                                                  args.state_size,
                                                  device=device)
        # Update belief/state using posterior from previous belief/state, previous action and current observation (over entire sequence at once)
        beliefs, prior_states, prior_means, prior_std_devs, posterior_states, posterior_means, posterior_std_devs = transition_model(
            init_state, actions[:-1], init_belief,
            bottle(encoder, (observations[1:], )), nonterminals[:-1])
        # Calculate observation likelihood, reward likelihood and KL losses (for t = 0 only for latent overshooting); sum over final dims, average over batch and time (original implementation, though paper seems to miss 1/T scaling?)
        if args.worldmodel_LogProbLoss:
            observation_dist = Normal(
                bottle(observation_model, (beliefs, posterior_states)), 1)
            observation_loss = -observation_dist.log_prob(
                observations[1:]).sum(dim=(2, 3, 4)).mean(dim=(0, 1))
        else:
            observation_loss = F.mse_loss(
                bottle(observation_model, (beliefs, posterior_states)),
                observations[1:],
                reduction='none').sum(dim=(2, 3, 4)).mean(dim=(0, 1))
        if args.worldmodel_LogProbLoss:
            reward_dist = Normal(
                bottle(reward_model, (beliefs, posterior_states)), 1)
            reward_loss = -reward_dist.log_prob(rewards[:-1]).mean(dim=(0, 1))
        else:
            reward_loss = F.mse_loss(bottle(reward_model,
                                            (beliefs, posterior_states)),
                                     rewards[:-1],
                                     reduction='none').mean(dim=(0, 1))
        # pcont related
        pcont_loss = 0.0
        if args.pcont:
            pcont_pred = Bernoulli(bottle(pcont_model, (beliefs, posterior_states)))
            pcont_target = args.discount * nonterminals[:-1].squeeze(-1)
            pcont_loss += -torch.mean(pcont_pred.log_prob(pcont_target))
            pcont_loss *= args.pcont_scale


        # transition loss
        div = kl_divergence(Normal(posterior_means, posterior_std_devs),
                            Normal(prior_means, prior_std_devs)).sum(dim=2)
        kl_loss = args.kl_scale * torch.max(div, free_nats).mean(dim=(0, 1))
        if args.global_kl_beta != 0:
            kl_loss += args.global_kl_beta * kl_divergence(
                Normal(posterior_means, posterior_std_devs),
                global_prior).sum(dim=2).mean(dim=(0, 1))
        # Apply linearly ramping learning rate schedule
        if args.learning_rate_schedule != 0:
            for group in model_optimizer.param_groups:
                group['lr'] = min(
                    group['lr'] + args.model_learning_rate /
                    args.model_learning_rate_schedule,
                    args.model_learning_rate)
        model_loss = observation_loss + reward_loss + kl_loss + pcont_loss
        # Update model parameters
        model_optimizer.zero_grad()
        model_loss.backward()
        nn.utils.clip_grad_norm_(param_list, args.grad_clip_norm, norm_type=2)
        model_optimizer.step()

        #Dreamer implementation: actor loss calculation and optimization
        with torch.no_grad():
            actor_states = posterior_states.detach()
            actor_beliefs = beliefs.detach()
        with FreezeParameters(model_modules):
            imagination_traj = imagine_ahead(args, actor_states, actor_beliefs,
                                             actor_model, transition_model)
        imged_beliefs, imged_prior_states, imged_prior_means, imged_prior_std_devs = imagination_traj
        with FreezeParameters(model_modules + value_model.modules):
            imged_reward = bottle(reward_model,
                                  (imged_beliefs, imged_prior_states))
            value_pred = bottle(value_model,
                                (imged_beliefs, imged_prior_states))
            if args.pcont:
                pcont = Bernoulli(bottle(pcont_model, (imged_beliefs, imged_prior_states))).mean
            else:
                pcont = args.discount * torch.ones_like(imged_reward)

        returns = lambda_return(
            imged_reward[:-1], value_pred[:-1], pcont[:-1],
            bootstrap=value_pred[-1], lambda_=args.disclam)
        discount = torch.cat([torch.ones_like(pcont[:1]), pcont[:-2]])
        discount = torch.cumprod(discount, 0).detach()
        actor_loss = -torch.mean(discount * returns)

        # Update model parameters
        actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(actor_model.parameters(),
                                 args.grad_clip_norm,
                                 norm_type=2)
        actor_optimizer.step()

        #Dreamer implementation: value loss calculation and optimization
        with torch.no_grad():
            value_beliefs = imged_beliefs.detach()
            value_prior_states = imged_prior_states.detach()
            target = returns.detach()
        value_dist = Normal(
            bottle(value_model, (value_beliefs, value_prior_states))[:-1], 1) 
        value_loss = -(discount * value_dist.log_prob(target)).mean(dim=(0, 1))
        
        # Update model parameters
        value_optimizer.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad_norm_(value_model.parameters(),
                                 args.grad_clip_norm,
                                 norm_type=2)
        value_optimizer.step()

        # # Store (0) observation loss (1) reward loss (2) KL loss (3) actor loss (4) value loss
        losses.append([
            observation_loss.item(),
            reward_loss.item(),
            kl_loss.item(),
            actor_loss.item(),
            value_loss.item()
        ])

    # Update and plot loss metrics
    losses = tuple(zip(*losses))
    metrics['observation_loss'].append(losses[0])
    metrics['reward_loss'].append(losses[1])
    metrics['kl_loss'].append(losses[2])
    metrics['actor_loss'].append(losses[3])
    metrics['value_loss'].append(losses[4])
    lineplot(metrics['episodes'][-len(metrics['observation_loss']):],
             metrics['observation_loss'], 'observation_loss', results_dir)
    lineplot(metrics['episodes'][-len(metrics['reward_loss']):],
             metrics['reward_loss'], 'reward_loss', results_dir)
    lineplot(metrics['episodes'][-len(metrics['kl_loss']):],
             metrics['kl_loss'], 'kl_loss', results_dir)
    lineplot(metrics['episodes'][-len(metrics['actor_loss']):],
             metrics['actor_loss'], 'actor_loss', results_dir)
    lineplot(metrics['episodes'][-len(metrics['value_loss']):],
             metrics['value_loss'], 'value_loss', results_dir)

    # Data collection
    print("Data collection")
    with torch.no_grad():
        observation, total_reward = env.reset(), 0
        belief, posterior_state, action = torch.zeros(
            1, args.belief_size,
            device=device), torch.zeros(1,
                                             args.state_size,
                                             device=device), torch.zeros(
                                                 1,
                                                 env.action_size,
                                                 device=device)
        pbar = tqdm(range(args.max_episode_length // args.action_repeat))
        for t in pbar:
            # print("step",t)
            belief, posterior_state, action, next_observation, reward, done = update_belief_and_act(
                args,
                env,
                planner,
                transition_model,
                encoder,
                belief,
                posterior_state,
                action,
                observation.to(device),
                explore=True,
                step=t + metrics['steps'][-1])
            D.append(observation, action.cpu(), reward, done)
            total_reward += reward
            observation = next_observation
            if args.render:
                env.render()
            if done:
                pbar.close()
                break
        if not done and args.max_episode_length<10000:
            args.max_episode_length+=100

        # Update and plot train reward metrics
        metrics['steps'].append(t + metrics['steps'][-1])
        metrics['episodes'].append(episode)
        metrics['train_rewards'].append(total_reward)
        lineplot(metrics['episodes'][-len(metrics['train_rewards']):],
                 metrics['train_rewards'], 'train_rewards', results_dir)

    # Test model
    print("Test model")
    if episode % args.test_interval == 0:
        # Set models to eval mode
        transition_model.eval()
        observation_model.eval()
        reward_model.eval()
        encoder.eval()
        actor_model.eval()
        value_model.eval()
        if args.pcont:
            pcont_model.eval()
        # Initialise parallelised test environments

        with torch.no_grad():
            observation, total_rewards, video_frames = test_envs.reset(
            ), np.zeros((args.test_episodes, )), []
            belief, posterior_state, action = torch.zeros(
                args.test_episodes, args.belief_size,
                device=device), torch.zeros(
                    args.test_episodes, args.state_size,
                    device=device), torch.zeros(args.test_episodes,
                                                     env.action_size,
                                                     device=device)
            pbar = tqdm(range(args.max_episode_length // args.action_repeat))
            for t in pbar:
                belief, posterior_state, action, next_observation, reward, done = update_belief_and_act(
                    args, test_envs, planner, transition_model, encoder,
                    belief, posterior_state, action,
                    observation.to(device))
                total_rewards += reward.numpy()
                video_frames.append(
                    make_grid(torch.cat([
                        observation,
                        observation_model(belief, posterior_state).cpu()
                    ],
                                        dim=3) + 0.5,
                              nrow=5).numpy())  # Decentre
                observation = next_observation
                if done.sum().item() == args.test_episodes:
                    pbar.close()
                    break

        # Update and plot reward metrics (and write video if applicable) and save metrics
        metrics['test_episodes'].append(episode)
        metrics['test_rewards'].append(total_rewards.tolist())
        lineplot(metrics['test_episodes'], metrics['test_rewards'],
                 'test_rewards', results_dir)
        lineplot(np.asarray(
            metrics['steps'])[np.asarray(metrics['test_episodes']) - 1],
                 metrics['test_rewards'],
                 'test_rewards_steps',
                 results_dir,
                 xaxis='step')
        episode_str = str(episode).zfill(len(str(args.episodes)))
        write_video(video_frames, 'test_episode_%s' % episode_str,
                    results_dir)  # Lossy compression
        save_image(
            torch.as_tensor(video_frames[-1]),
            os.path.join(results_dir, 'test_episode_%s.png' % episode_str))
        torch.save(metrics, os.path.join(results_dir, 'metrics.pth'))

        # Set models to train mode
        transition_model.train()
        observation_model.train()
        reward_model.train()
        encoder.train()
        actor_model.train()
        value_model.train()
        if args.pcont:
            pcont_model.train()
        # Close test environments
        # test_envs.close()

    writer.add_scalar("train_reward", metrics['train_rewards'][-1],
                      metrics['steps'][-1])
    writer.add_scalar("train/episode_reward", metrics['train_rewards'][-1],
                      metrics['steps'][-1] * args.action_repeat)
    writer.add_scalar("observation_loss", metrics['observation_loss'][0][-1],
                      metrics['steps'][-1])
    writer.add_scalar("reward_loss", metrics['reward_loss'][0][-1],
                      metrics['steps'][-1])
    writer.add_scalar("kl_loss", metrics['kl_loss'][0][-1],
                      metrics['steps'][-1])
    writer.add_scalar("actor_loss", metrics['actor_loss'][0][-1],
                      metrics['steps'][-1])
    writer.add_scalar("value_loss", metrics['value_loss'][0][-1],
                      metrics['steps'][-1])
    print("episodes: {}, total_steps: {}, train_reward: {} ".format(
        metrics['episodes'][-1], metrics['steps'][-1],
        metrics['train_rewards'][-1]))

    # Checkpoint models
    if episode % args.checkpoint_interval == 0:
        state_dict = {
            'transition_model': transition_model.state_dict(), 
            'observation_model': observation_model.state_dict(), 
            'reward_model': reward_model.state_dict(), 
            'encoder': encoder.state_dict(), 
            'actor_model': actor_model.state_dict(), 
            'value_model': value_model.state_dict(), 
            'model_optimizer': model_optimizer.state_dict(), 
            'actor_optimizer': actor_optimizer.state_dict(), 
            'value_optimizer': value_optimizer.state_dict()
        }
        if args.pcont:
            state_dict['pcont_model'] = pcont_model.state_dict()
        torch.save(state_dict, os.path.join(results_dir, 'models_%d.pth' % episode))
        if args.checkpoint_experience:
            torch.save(
                D, os.path.join(results_dir, 'experience.pth')
            )  # Warning: will fail with MemoryError with large memory sizes

# Close training environment
env.close()
test_envs.close()