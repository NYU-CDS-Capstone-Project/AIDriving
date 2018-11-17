import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


class RolloutStorage(object):
    def __init__(self, num_steps, num_processes, obs_shape, action_space, state_size):
        self.observations = torch.zeros(num_steps + 1, num_processes, *obs_shape)
        self.states = torch.zeros(num_steps + 1, num_processes, state_size)
        self.rewards = torch.zeros(num_steps, num_processes, 1)
        self.value_preds = torch.zeros(num_steps + 1, num_processes, 1)
        self.returns = torch.zeros(num_steps + 1, num_processes, 1)
        self.discrete_action_log_probs = torch.zeros(num_steps, num_processes, 1)
        self.continuous_action_log_probs = torch.zeros(num_steps, num_processes, 1)
        self.discrete_actions = torch.zeros(num_steps, num_processes, 1).long()
        self.continuous_actions = torch.zeros(num_steps, num_processes, action_space.shape[0])
        self.masks = torch.ones(num_steps + 1, num_processes, 1)

    def cuda(self):
        self.observations = self.observations.cuda()
        self.states = self.states.cuda()
        self.rewards = self.rewards.cuda()
        self.value_preds = self.value_preds.cuda()
        self.returns = self.returns.cuda()
        self.continuous_action_log_probs = self.continuous_action_log_probs.cuda()
        self.discrete_action_log_probs = self.discrete_action_log_probs.cuda()
        self.discrete_actions = self.discrete_actions.cuda()
        self.continuous_actions = self.continuous_actions.cuda()
        self.masks = self.masks.cuda()

    def insert(self, step, current_obs, state, discrete_action, discrete_action_log_prob, continuous_action, continuous_action_log_prob, value_pred, reward, mask):
        self.observations[step + 1].copy_(current_obs)
        self.states[step + 1].copy_(state)

        self.discrete_actions[step].copy_(discrete_action)
        self.discrete_action_log_probs[step].copy_(discrete_action_log_prob)
        self.continuous_actions[step].copy_(continuous_action)
        self.continuous_action_log_probs[step].copy_(continuous_action_log_prob)

        self.value_preds[step].copy_(value_pred)
        self.rewards[step].copy_(reward)
        self.masks[step + 1].copy_(mask)

    def after_update(self):
        self.observations[0].copy_(self.observations[-1])
        self.states[0].copy_(self.states[-1])
        self.masks[0].copy_(self.masks[-1])

    def compute_returns(self, next_value, use_gae, gamma, tau):
        if use_gae:
            self.value_preds[-1] = next_value
            gae = 0
            for step in reversed(range(self.rewards.size(0))):
                delta = self.rewards[step] + gamma * self.value_preds[step + 1] * self.masks[step + 1] - self.value_preds[step]
                gae = delta + gamma * tau * self.masks[step + 1] * gae
                self.returns[step] = gae + self.value_preds[step]
        else:
            self.returns[-1] = next_value
            for step in reversed(range(self.rewards.size(0))):
                self.returns[step] = self.returns[step + 1] * \
                    gamma * self.masks[step + 1] + self.rewards[step]
