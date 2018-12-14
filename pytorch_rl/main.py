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

    envs = make_env_vec(args.num_processes, args.env_name, args.seed, args.log_dir, args.start_container, max_steps = 1200, discrete_wrapper=args.discrete_actions)

    action_shape = 1 if envs.action_space.__class__.__name__ == "Discrete" else envs.action_space.shape[0]
    obs_shape = (envs.observation_space.shape[0] * args.num_stack, *envs.observation_space.shape[1:])
    distribution = 'MixedDistribution' if args.use_mixed else 'DiagGaussian'
    actor_critic = CNNPolicy(obs_shape[0], envs.action_space, args.recurrent_policy, args.use_vae, args.only_ae, 
        args.use_batchnorm, args.use_residual, args.policy_backprop, distribution=distribution)
    print_model_size(actor_critic)
    print(obs_shape)
    if args.cuda: actor_critic.cuda()

    if args.algo == 'a2c':
        optimizer = optim.RMSprop(actor_critic.parameters(), args.lr, eps=args.eps, alpha=args.alpha)

    rollouts = RolloutStorage(args.num_steps, args.num_processes, obs_shape, envs.action_space, actor_critic.state_size, actor_critic.epsilon_size)
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
        '''if j%200 == 0:
            envs.envs[0].seed(args.seed)
            obs = envs.reset()
            current_obs = update_current_obs(obs, current_obs, envs.observation_space.shape[0], args.num_stack)
            rollouts.observations[0].copy_(current_obs)'''

        for step in range(args.num_steps):
            # Sample actions
            value, action, action_log_prob, states, epsilons, recon = actor_critic.act(
                Variable(rollouts.observations[step]),
                Variable(rollouts.states[step]),
                Variable(rollouts.masks[step])
            )

            '''if j%200 == 0 and step%5 == 0:
                print(current_obs)
                print(recon)'''

            cpu_actions = action.data.squeeze(1).cpu().numpy()
            # Exploration epsilon greedy, ToDo better exploration policy
            if np.random.random_sample() < args.exp_probability:
                cpu_actions = [envs.action_space.sample() for _ in range(args.num_processes)]

            # Observation, reward and next obs
            obs, reward, done, info = envs.step(cpu_actions)

            #ToDo better collision strategy
            '''for i, flag in enumerate(done):
                if flag == True:
                    envs.envs[i].user_tile_start = info[i]['Simulator']['tile_coords']
                    envs.envs[i].reset()
                    envs.envs[i].user_tile_start = None'''

            # Maxime: clip the reward within [0,1] for more reliable training
            # This code deals poorly with large reward values
            #reward = np.clip(reward, a_min=0, a_max=None) / 400

            slack = args.reward_slack
            scaled_reward = np.clip(reward + slack, a_min = -2.0**args.reward_pow, a_max=None)
            for i in range(args.num_processes):
            	if scaled_reward[i] > 0: scaled_reward[i] = (1 + scaled_reward[i])**args.reward_pow - 1
            scaled_reward = torch.from_numpy(np.expand_dims(np.stack(scaled_reward), 1)).float()

            '''if step != 0:
                cur_angle = action[:, 1]
                scaled_reward -= torch.abs(prev_angle - cur_angle).view(-1).data.cpu()*args.reward_factor
            prev_angle = action[:, 1]'''

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
            rollouts.insert(step, current_obs, states.data, action.data, action_log_prob.data, value.data, scaled_reward, masks, epsilons.data)

        next_value = actor_critic(
            Variable(rollouts.observations[-1]),
            Variable(rollouts.states[-1]),
            Variable(rollouts.masks[-1])
        )[0].data

        rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.tau)

        #Performing Actor Critic Updates
        if args.algo in ['a2c']:

            recurrence_steps = 1
            if args.recurrent_policy: recurrence_steps = args.num_recsteps

            indices = torch.arange(0, args.num_steps, recurrence_steps).long()
            if args.cuda: indices = indices.cuda()

            total_value_loss = 0
            total_action_loss = 0
            total_dist_entropy = 0
            total_loss = 0
            total_recon_loss = 0
            total_kdl_loss = 0

            for rstep in range(recurrence_steps):

                values, action_log_probs, dist_entropy, states, recon_loss, kld = actor_critic.evaluate_actions(
                    Variable(rollouts.observations[:-1].index_select(0, indices).view(-1, *obs_shape)),
                    Variable(rollouts.states[:-1].index_select(0, indices).view(-1, actor_critic.state_size)),
                    Variable(rollouts.masks[:-1].index_select(0, indices).view(-1, 1)),
                    Variable(rollouts.actions.index_select(0, indices).view(-1, action_shape)),
                    Variable(rollouts.epsilons.index_select(0, indices).view(-1, actor_critic.epsilon_size)) if args.use_vae and not args.only_ae else None
                )

                values = values.view(int(args.num_steps/recurrence_steps), args.num_processes, 1)
                action_log_probs = action_log_probs.view(int(args.num_steps/recurrence_steps), args.num_processes, 1)

                advantages = Variable(rollouts.returns[:-1].index_select(0, indices)) - values
                value_loss = advantages.pow(2).mean()

                action_loss = -(Variable(advantages.data) * action_log_probs).mean()

                k = args.vae_coef
                
                if j > args.vae_steps:
                    loss = value_loss * args.value_loss_coef + action_loss - dist_entropy * args.entropy_coef + k*recon_loss + 0.5*k*kld
                else: loss = recon_loss + 0.05*kld

                total_value_loss += value_loss.data[0]
                total_action_loss += action_loss.data[0]
                total_dist_entropy += dist_entropy.data[0]
                if args.use_vae:
                    total_recon_loss += recon_loss.data[0]
                    if not args.only_ae: total_kdl_loss += kld.data[0]
                total_loss += loss

                indices += 1

            total_value_loss /= recurrence_steps
            total_action_loss /= recurrence_steps
            total_dist_entropy /= recurrence_steps
            total_recon_loss /= recurrence_steps
            total_kdl_loss /= recurrence_steps
            total_loss /= recurrence_steps

            optimizer.zero_grad()
            total_loss.backward()

            if args.algo == 'a2c':
                nn.utils.clip_grad_norm(actor_critic.parameters(), args.max_grad_norm)

            optimizer.step()

        rollouts.after_update()

        #Saving the model
        if j % args.save_interval == 0 and args.save_dir != "":
            logger.save_model(args.save_dir, actor_critic, envs, args.algo, args.env_name, args.name, args.cuda)

        #Logging the model
        if j % args.log_interval == 0:
            logger.print_log(total_value_loss, total_action_loss, total_dist_entropy, total_recon_loss, total_kdl_loss, 
                args.num_processes, args.num_steps, j, start)

if __name__ == "__main__":
    main()
