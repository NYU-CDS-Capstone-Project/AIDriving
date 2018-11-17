import math

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

        action_mean_left[:, 0] = 0.2 + torch.nn.functional.sigmoid(action_mean_left[:, 0]) / 2.5
        action_mean_left[:, 1] = 0.4 + torch.nn.functional.sigmoid(action_mean_left[:, 1]) * 3/5

        action_mean_right[:, 0] = 0.2 + torch.nn.functional.sigmoid(action_mean_right[:, 0]) / 2.5
        action_mean_right[:, 1] = - 0.4 - torch.nn.functional.sigmoid(action_mean_right[:, 1]) * 3/5

        action_mean_straight[:, 0] = 0.6 + torch.nn.functional.sigmoid(action_mean_left[:, 0]) / 2.5
        action_mean_straight[:, 1] = torch.nn.functional.tanh(action_mean_straight[:, 1]) * 1/5

        direction = self.linear(x)

        action_logstd = Variable(torch.log(torch.sqrt(torch.Tensor([0.5])))).repeat(self.num_outputs)

        return direction, action_mean_left, action_mean_right, action_mean_straight, action_logstd

    def sample(self, x, deterministic):

        direction, action_mean_left, action_mean_right, action_mean_straight, action_logstd = self(x)
        action_std = action_logstd.exp()

        direction_probs = F.softmax(direction, dim=1)
        direction_max = torch.max(direction_probs, dim=1)[0].unsqueeze(-1)
        direction_onehot = torch.eq(direction_probs, direction_max).unsqueeze(-1)
        #direction_indices = torch.argmax(direction_probs, dim=1)

        action_mean_left = torch.unsqueeze(action_mean_left, -1)
        action_mean_right = torch.unsqueeze(action_mean_right, -1)
        action_mean_straight = torch.unsqueeze(action_mean_straight, -1)

        action_mean = torch.cat([action_mean_left, action_mean_right, action_mean_straight], dim=2).permute(0,2,1)
        #action_mean = torch.index_select(action_mean, 2, direction_indices).squeeze(-1)

        action_mean = action_mean*direction_onehot.float()
        action_mean = action_mean.sum(dim=1)

        if deterministic is False:
            noise = Variable(torch.randn(action_std.size()))
            if action_std.is_cuda:
                noise = noise.cuda()
            action = action_mean + action_std * noise
        else:
            action = action_mean

        return action

    def logprobs_and_entropy(self, x, actions):
        direction, action_mean_left, action_mean_right, action_mean_straight, action_logstd = self(x)
        action_std = action_logstd.exp()

        direction_probs = F.softmax(direction, dim=1)
        direction_max = torch.max(direction_probs, dim=1)[0].unsqueeze(-1)
        direction_onehot = torch.eq(direction_probs, direction_max).unsqueeze(-1)
        #direction_indices = torch.argmax(direction_probs, dim=1)

        action_mean_left = torch.unsqueeze(action_mean_left, -1)
        action_mean_right = torch.unsqueeze(action_mean_right, -1)
        action_mean_straight = torch.unsqueeze(action_mean_straight, -1)

        action_mean = torch.cat([action_mean_left, action_mean_right, action_mean_straight], dim=2).permute(0,2,1)
        #action_mean = torch.index_select(action_mean, 2, direction_indices).squeeze(-1)

        action_mean = action_mean*direction_onehot.float()
        action_mean = action_mean.sum(dim=1)

        action_log_probs = -0.5 * ((actions - action_mean) / action_std).pow(2) - 0.5 * math.log(2 * math.pi) - action_logstd
        action_log_probs = action_log_probs.sum(-1, keepdim=True)
        dist_entropy = 0.5 + 0.5 * math.log(2 * math.pi) + action_logstd
        dist_entropy = dist_entropy.sum(-1).mean()
        return action_log_probs, dist_entropy

