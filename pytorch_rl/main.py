import copy
import glob
import os
import time

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from arguments import get_args
from envs import make_env_vec
from utils import print_model_size, update_current_obs
from model import CNNPolicy, MLPPolicy
from storage import RolloutStorage
from logger import ModelLogger

args = get_args()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


def main():
    os.environ['OMP_NUM_THREADS'] = '1'
    num_updates = int(args.num_frames) // args.num_steps // args.num_processes

    logger = ModelLogger(args.log_dir, args.num_processes)

    envs = make_env_vec(args.num_processes, args.env_name, args.seed, args.log_dir, args.start_container, max_steps = 1200)

    action_shape = 1 if envs.action_space.__class__.__name__ == "Discrete" else envs.action_space.shape[0]
    obs_shape = (envs.observation_space.shape[0] * args.num_stack, *envs.observation_space.shape[1:])
    actor_critic = CNNPolicy(obs_shape[0], envs.action_space, args.recurrent_policy)
    print_model_size(actor_critic)
    if args.cuda: actor_critic.cuda()

    if args.algo == 'a2c':
        optimizer = optim.RMSprop(actor_critic.parameters(), args.lr, eps=args.eps, alpha=args.alpha)

    rollouts = RolloutStorage(args.num_steps, args.num_processes, obs_shape, envs.action_space, actor_critic.state_size)
    current_obs = torch.zeros(args.num_processes, *obs_shape)
    obs = envs.reset()
    current_obs = update_current_obs(obs, current_obs, envs.observation_space.shape[0], args.num_stack)
    rollouts.observations[0].copy_(current_obs)

    if args.cuda:
        current_obs = current_obs.cuda()
        rollouts.cuda()

    start = time.time()
    for j in range(num_updates):
        #Running an episode
        for step in range(args.num_steps):
            # Sample actions
            value, action, action_log_prob, states = actor_critic.act(
                Variable(rollouts.observations[step]),
                Variable(rollouts.states[step]),
                Variable(rollouts.masks[step])
            )
            cpu_actions = action.data.squeeze(1).cpu().numpy()
            # Exploration epsilon greedy
            if np.random.random_sample() < 0.2:
                cpu_actions = [envs.action_space.sample() for _ in range(args.num_processes)]

            # Observation, reward and next obs
            obs, reward, done, info = envs.step(cpu_actions)

            # Maxime: clip the reward within [0,1] for more reliable training
            # This code deals poorly with large reward values
            #reward = np.clip(reward, a_min=0, a_max=None) / 400

            scaled_reward = np.clip(reward + 0.4, a_min = -3.0, a_max=None)            
            scaled_reward = torch.from_numpy(np.expand_dims(np.stack(scaled_reward), 1)).float()

            reward = np.clip(reward, a_min=-4.0, a_max=None) + 1.0
            reward = torch.from_numpy(np.expand_dims(np.stack(reward), 1)).float()

            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])

            logger.update_reward(reward, masks)

            if args.cuda:
                masks = masks.cuda()

            if current_obs.dim() == 4:
                current_obs *= masks.unsqueeze(2).unsqueeze(2)
            else:
                current_obs *= masks

            current_obs = update_current_obs(obs, current_obs, envs.observation_space.shape[0], args.num_stack)
            rollouts.insert(step, current_obs, states.data, action.data, action_log_prob.data, value.data, scaled_reward, masks)

        next_value = actor_critic(
            Variable(rollouts.observations[-1]),
            Variable(rollouts.states[-1]),
            Variable(rollouts.masks[-1])
        )[0].data

        rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.tau)

        #Performing Actor Critic Updates
        if args.algo in ['a2c']:
            values, action_log_probs, dist_entropy, states = actor_critic.evaluate_actions(Variable(rollouts.observations[:-1].view(-1, *obs_shape)),
                                                                                           Variable(rollouts.states[0].view(-1, actor_critic.state_size)),
                                                                                           Variable(rollouts.masks[:-1].view(-1, 1)),
                                                                                           Variable(rollouts.actions.view(-1, action_shape)))

            values = values.view(args.num_steps, args.num_processes, 1)
            action_log_probs = action_log_probs.view(args.num_steps, args.num_processes, 1)

            advantages = Variable(rollouts.returns[:-1]) - values
            value_loss = advantages.pow(2).mean()

            action_loss = -(Variable(advantages.data) * action_log_probs).mean()

            optimizer.zero_grad()
            (value_loss * args.value_loss_coef + action_loss - dist_entropy * args.entropy_coef).backward()

            if args.algo == 'a2c':
                nn.utils.clip_grad_norm(actor_critic.parameters(), args.max_grad_norm)

            optimizer.step()

        rollouts.after_update()

        #Saving the model
        if j % args.save_interval == 0 and args.save_dir != "":
            logger.save_model(args.save_dir, actor_critic, envs, args.algo, args.env_name, args.name, args.cuda)

        #Logging the model
        if j % args.log_interval == 0:
            logger.print_log(value_loss, action_loss, dist_entropy, args.num_processes, args.num_steps, j, start)

if __name__ == "__main__":
    main()
