import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import AddBias
from arguments import get_args

args = get_args()

class Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs, ):
        super(Categorical, self).__init__()
        self.linear = nn.Linear(num_inputs, num_outputs)

    def forward(self, x):
        x = self.linear(x)
        return x

    def sample(self, x, deterministic):
        x = self(x)

        probs = F.softmax(x, dim=1)
        if deterministic is False:
            action = probs.multinomial(num_samples=1)
        else:
            action = probs.max(1, keepdim=True)[1]
        return action

    def logprobs_and_entropy(self, x, actions):
        x = self(x)

        log_probs = F.log_softmax(x, dim=1)
        probs = F.softmax(x, dim=1)

        action_log_probs = log_probs.gather(1, actions)

        dist_entropy = -(log_probs * probs).sum(-1).mean()
        return action_log_probs, dist_entropy


class DiagGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(DiagGaussian, self).__init__()
        self.fc_mean = nn.Linear(num_inputs, num_outputs)
        self.logstd = AddBias(torch.zeros(num_outputs))
        self.num_outputs = num_outputs

    def forward(self, x):
        action_mean = self.fc_mean(x)
        action_mean[:, 0] = 0.4 + torch.nn.functional.sigmoid(action_mean[:, 0]) / 2.5
        action_mean[:, 1] = torch.nn.functional.tanh(action_mean[:, 1])

        #  An ugly hack for my KFAC implementation.
        zeros = Variable(torch.zeros(action_mean.size()), volatile=x.volatile)
        if x.is_cuda:
            zeros = zeros.cuda()

        action_logstd = Variable(torch.log(torch.sqrt(torch.Tensor([args.continuous_var])))).repeat(self.num_outputs)
        #action_logstd = Variable(torch.zeros(self.num_outputs))
        #action_logstd = self.logstd(zeros)
        if x.is_cuda:
            action_logstd = action_logstd.cuda()
        #action_logstd = torch.nn.functional.sigmoid(self.logstd(zeros)) / 3
        return action_mean, action_logstd

    def sample(self, x, deterministic):
        action_mean, action_logstd = self(x)

        action_std = action_logstd.exp()

        if deterministic is False:
            noise = Variable(torch.randn(action_std.size()))
            if action_std.is_cuda:
                noise = noise.cuda()
            action = action_mean + action_std * noise
        else:
            action = action_mean

        return action

    def logprobs_and_entropy(self, x, actions):
        action_mean, action_logstd = self(x)

        action_std = action_logstd.exp()

        action_log_probs = -0.5 * ((actions - action_mean) / action_std).pow(2) - 0.5 * math.log(2 * math.pi) - action_logstd
        action_log_probs = action_log_probs.sum(-1, keepdim=True)
        dist_entropy = 0.5 + 0.5 * math.log(2 * math.pi) + action_logstd
        dist_entropy = dist_entropy.sum(-1).mean()
        return action_log_probs, dist_entropy

class MixedDistribution(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(MixedDistribution, self).__init__()
        self.linear = nn.Linear(num_inputs, 3)
        self.fc_mean_left = nn.Linear(num_inputs, num_outputs)
        self.fc_mean_right = nn.Linear(num_inputs, num_outputs)
        self.fc_mean_straight = nn.Linear(num_inputs, num_outputs)
        self.num_outputs = num_outputs
    
    def forward(self, x):
        action_mean_left = self.fc_mean_left(x)
        action_mean_right = self.fc_mean_right(x)
        action_mean_straight = self.fc_mean_straight(x)

        action_mean_left[:, 0] = 0.4 + torch.nn.functional.sigmoid(action_mean_left[:, 0]) * 2/5
        action_mean_left[:, 1] = 0.6 + torch.nn.functional.sigmoid(action_mean_left[:, 1]) * 2/5

        action_mean_right[:, 0] = 0.4 + torch.nn.functional.sigmoid(action_mean_right[:, 0]) * 2/5
        action_mean_right[:, 1] = - 0.6 - torch.nn.functional.sigmoid(action_mean_right[:, 1]) * 2/5

        action_mean_straight[:, 0] = 0.7 + torch.nn.functional.sigmoid(action_mean_straight[:, 0]) * 3/10
        action_mean_straight[:, 1] = torch.nn.functional.tanh(action_mean_straight[:, 1]) * 1/2000

        direction = self.linear(x)

        action_logstd = Variable(torch.log(torch.sqrt(torch.Tensor([args.continuous_var for _ in range(self.num_outputs)]))))
        if x.is_cuda:
            action_logstd = action_logstd.cuda()

        return direction, action_mean_left, action_mean_right, action_mean_straight, action_logstd

    def sample(self, x, deterministic):

        direction, action_mean_left, action_mean_right, action_mean_straight, action_logstd = self(x)
        action_std = action_logstd.exp()

        if np.random.random_sample() < args.exp_probability:
            direction = Variable(torch.randn(direction.shape))
            if x.is_cuda:
                direction = direction.cuda()

        direction_probs = F.softmax(direction, dim=1)
        if deterministic is False:
            discrete_action = direction_probs.multinomial(num_samples=1)
        else:
            discrete_action = direction_probs.max(1, keepdim=True)[1]

        direction_onehot = Variable(torch.FloatTensor(discrete_action.size(0), discrete_action.size(1), 3).zero_())
        if x.is_cuda:
            direction_onehot = direction_onehot.cuda()

        direction_onehot.scatter_(2, discrete_action.unsqueeze(-1).long(), 1)
        direction_onehot = direction_onehot.squeeze(1).unsqueeze(-1)

        #direction_max = torch.max(direction_probs, dim=1)[0].unsqueeze(-1)
        #direction_onehot = torch.eq(direction_probs, direction_max).unsqueeze(-1)
        #direction_indices = torch.argmax(direction_probs, dim=1)

        action_mean_left = torch.unsqueeze(action_mean_left, -1)
        action_mean_right = torch.unsqueeze(action_mean_right, -1)
        action_mean_straight = torch.unsqueeze(action_mean_straight, -1)

        action_mean = torch.cat([action_mean_left, action_mean_right, action_mean_straight], dim=2).permute(0,2,1)
        #action_mean = torch.index_select(action_mean, 2, direction_indices).squeeze(-1)

        action_mean = action_mean*direction_onehot.float()
        action_mean = action_mean.sum(dim=1)

        if (discrete_action.long() == 2).any():
           action_std[1].data == 0.01

        if deterministic is False:
            noise = Variable(torch.randn(action_std.size()))
            if action_std.is_cuda:
                noise = noise.cuda()
            continous_action = action_mean + action_std * noise
        else:
            continous_action = action_mean

        return [continous_action, discrete_action]

    '''def logprobs_and_entropy(self, x, actions):
        continous_actions, discrete_actions = actions

        direction, action_mean_left, action_mean_right, action_mean_straight, action_logstd = self(x)
        action_std = action_logstd.exp()

        direction_probs = F.softmax(direction, dim=1)
        #direction_max, direction_argmax = torch.max(direction_probs, dim=1)
        #direction_onehot = torch.eq(direction_probs, direction_max.unsqueeze(-1)).unsqueeze(-1)
        direction_onehot = Variable(torch.FloatTensor(discrete_actions.size(0), discrete_actions.size(1), 3).zero_())
        if x.is_cuda:
            direction_onehot = direction_onehot.cuda()

        direction_onehot.scatter_(2, discrete_actions.unsqueeze(-1).long(), 1)
        direction_onehot = direction_onehot.squeeze(1).unsqueeze(-1)
        
        #action_mask = (direction_argmax.reshape(-1) == discrete_actions.reshape(-1)).reshape(-1,1)
        action_mean_left = torch.unsqueeze(action_mean_left, -1)
        action_mean_right = torch.unsqueeze(action_mean_right, -1)
        action_mean_straight = torch.unsqueeze(action_mean_straight, -1)

        action_mean = torch.cat([action_mean_left, action_mean_right, action_mean_straight], dim=2).permute(0,2,1)

        action_mean = action_mean*direction_onehot.float()
        action_mean = action_mean.sum(dim=1)

        continuous_action_log_probs = -0.5 * ((continous_actions - action_mean) / action_std).pow(2) - 0.5 * math.log(2 * math.pi) - action_logstd
        continuous_action_log_probs = continuous_action_log_probs.sum(-1, keepdim=True)
        continuous_dist_entropy = 0.5 + 0.5 * math.log(2 * math.pi) + action_logstd
        continuous_dist_entropy = continuous_dist_entropy.sum(-1).mean()

        #continuous_action_log_probs = continuous_action_log_probs*action_mask.float()
        #continuous_dist_entropy = continuous_dist_entropy*action_mask.float()

        direction_log_probs = F.log_softmax(direction, dim=1)
        discrete_action_log_probs = direction_log_probs.gather(1, discrete_actions)
        discrete_dist_entropy = -(direction_log_probs * direction_probs).sum(-1).mean()

        return [continuous_action_log_probs, discrete_action_log_probs], [continuous_dist_entropy, discrete_dist_entropy]'''

    def logprobs_and_entropy(self, x, actions):
        continous_actions, discrete_actions = actions

        direction, action_mean_left, action_mean_right, action_mean_straight, action_logstd = self(x)
        action_std = action_logstd.exp()

        direction_probs = F.softmax(direction, dim=1)

        action_mean_left_probs = (1/(torch.sqrt(2*math.pi*action_std.pow(2))))*torch.exp(-0.5 * ((continous_actions - action_mean_left) / action_std).pow(2))
        action_mean_right_probs = (1/(torch.sqrt(2*math.pi*action_std.pow(2))))*torch.exp(-0.5 * ((continous_actions - action_mean_right) / action_std).pow(2))
        action_mean_straight_probs = (1/(torch.sqrt(2*math.pi*action_std.pow(2))))*torch.exp(-0.5 * ((continous_actions - action_mean_straight) / action_std).pow(2))

        action_mean_probs = torch.stack([action_mean_left_probs, action_mean_right_probs, action_mean_straight_probs]).permute(1, 0, 2)

        direction_probs = direction_probs.unsqueeze(1)
        continuous_mixed_probs = torch.bmm(direction_probs, action_mean_probs)

        continuous_mixed_probs = torch.log(continuous_mixed_probs)

        direction_log_probs = F.log_softmax(direction, dim=1)
        discrete_action_log_probs = direction_log_probs.gather(1, discrete_actions)

        continuous_dist_entropy = None
        discrete_dist_entropy = None

        continuous_mixed_probs = continuous_mixed_probs.squeeze(1).sum(-1, keepdim=True)
        return [continuous_mixed_probs, discrete_action_log_probs], [continuous_dist_entropy, discrete_dist_entropy]


