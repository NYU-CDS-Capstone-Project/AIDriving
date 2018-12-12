import operator
from functools import reduce
import torch
import torch.nn as nn
import torch.nn.functional as F
from distributions import Categorical, DiagGaussian, MixedDistribution
from utils import orthogonal
from torch.autograd import Variable

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        orthogonal(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0)


class FFPolicy(nn.Module):
    def __init__(self):
        super(FFPolicy, self).__init__()

    def forward(self, inputs, states, masks):
        raise NotImplementedError

    def act(self, inputs, states, masks, deterministic=False):
        value, x, states, _, _, _ = self(inputs, states, masks)
        action = self.dist.sample(x, deterministic=deterministic)
        action_log_probs, dist_entropy = self.dist.logprobs_and_entropy(x, action)
        return value, action, action_log_probs, states

    def evaluate_actions(self, inputs, states, masks, actions):
        value, x, states, reconstruction, mu, logvar = self(inputs, states, masks)
        action_log_probs, dist_entropy = self.dist.logprobs_and_entropy(x, actions)
        reconstruction_loss, kl_divergence = 0.0, 0.0
        if self.use_vae:
            reconstruction_loss, kl_divergence = self.compute_vae_loss(inputs, reconstruction, mu, logvar)
        return value, action_log_probs, dist_entropy, states, reconstruction_loss, kl_divergence


class CNNBlock(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size, stride, padding, use_batch_norm, use_residual):
        super(CNNBlock, self).__init__()
        self.use_batch_norm = use_batch_norm
        self.use_residual = use_residual

        self.conv = nn.Conv2d(channels_in, channels_out, kernel_size, stride=stride, padding=padding)
        self.conv_drop = torch.nn.Dropout2d(p=0.2)
        if use_batch_norm:
            self.conv_bn = torch.nn.BatchNorm2d(channels_out)
        self.reset_parameters()

    def reset_parameters(self):
        self.apply(weights_init)
        relu_gain = nn.init.calculate_gain('leaky_relu')
        self.conv.weight.data.mul_(relu_gain)

    def forward(self, inputs):
        x = self.conv(inputs)
        #x = self.conv_drop(x)
        if self.use_batch_norm:
            x = self.conv_bn(x)
        if self.use_residual:
            x = x + inputs
        x = F.leaky_relu(x)
        return x

class CNNUnBlock(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size, stride, padding, outpadding, use_batch_norm, use_residual):
        super(CNNUnBlock, self).__init__()
        self.use_batch_norm = use_batch_norm
        self.use_residual = use_residual

        self.unconv = nn.ConvTranspose2d(channels_out, channels_in, kernel_size, stride=stride, padding=padding, output_padding=outpadding)
        self.unconv_drop = torch.nn.Dropout2d(p=0.2)
        if use_batch_norm:
            self.unconv_bn = torch.nn.BatchNorm2d(channels_in)
        self.reset_parameters()

    def reset_parameters(self):
        self.apply(weights_init)
        relu_gain = nn.init.calculate_gain('leaky_relu')
        self.unconv.weight.data.mul_(relu_gain)

    def forward(self, inputs, last=False):
        x = self.unconv(inputs)
        #x = self.conv_drop(x)
        if self.use_batch_norm and not last:
            x = self.unconv_bn(x)
        if self.use_residual and not last:
            x = x + inputs
        if last: x = F.sigmoid(x)
        else: x = F.relu(x)
        return x


class CNNPolicy(FFPolicy):
    def __init__(self, num_inputs, action_space, use_gru, use_vae, 
        use_batch_norm=False, use_residual=False, distribution = 'DiagGaussian'):

        super(CNNPolicy, self).__init__()
        print('num_inputs=%s' % str(num_inputs))

        self.conv1 = CNNBlock(num_inputs, 32, 7, 2, 3, use_batch_norm, False)
        self.conv2 = CNNBlock(32, 32, 3, 2, 1, use_batch_norm, use_residual)
        self.conv3 = CNNBlock(32, 32, 3, 2, 1, use_batch_norm, use_residual)
        self.conv4 = CNNBlock(32, 32, 3, 2, 1, use_batch_norm, use_residual)
        self.conv5 = CNNBlock(32, 32, 3, 1, 1, use_batch_norm, use_residual)

        self.linear1_drop = nn.Dropout(p=0.3)
        self.linear1 = nn.Linear(32 * 10 * 8, 256)

        self.gruhdim = 256
        if use_gru:
            self.gru = nn.GRUCell(self.gruhdim, self.gruhdim)

        self.use_vae = use_vae
        if use_vae:
            self.linearmean = nn.Linear(256, 256)
            self.linearvar = nn.Linear(256, 256)
            self.unlinearlatent = nn.Linear(256, 256)
            self.unlinear1 = nn.Linear(256, 32 * 10 * 8)

            self.unconv5 = CNNUnBlock(32, 32, 3, 1, 1, 0, use_batch_norm, use_residual)
            self.unconv4 = CNNUnBlock(32, 32, 3, 2, (1,1), (1,0), use_batch_norm, use_residual)
            self.unconv3 = CNNUnBlock(32, 32, 3, 2, 1, 1, use_batch_norm, use_residual)
            self.unconv2 = CNNUnBlock(32, 32, 3, 2, 1, 1, use_batch_norm, use_residual)
            self.unconv1 = CNNUnBlock(num_inputs, 32, 7, 2, 3, 1, False, False)

        self.critic_linear = nn.Linear(256, 1)

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(256, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            if distribution == 'DiagGaussian':
                self.dist = DiagGaussian(256, num_outputs)
            elif distribution == 'MixedDistribution':
                self.dist = MixedDistribution(256, num_outputs)
        else:
            raise NotImplementedError

        self.train()
        self.reset_parameters()

    @property
    def state_size(self):
        if hasattr(self, 'gru'):
            return self.gruhdim
        else:
            return 1

    def reset_parameters(self):
        self.apply(weights_init)

        relu_gain = nn.init.calculate_gain('leaky_relu')
        self.linear1.weight.data.mul_(relu_gain)

        if hasattr(self, 'gru'):
            orthogonal(self.gru.weight_ih.data)
            orthogonal(self.gru.weight_hh.data)
            self.gru.bias_ih.data.fill_(0)
            self.gru.bias_hh.data.fill_(0)

        if self.dist.__class__.__name__ == "DiagGaussian":
            self.dist.fc_mean.weight.data.mul_(0.01)


    #scoure: https://github.com/coolvision/vae_conv/blob/master/vae_conv_model_mnist.py
    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)


    def forward(self, inputs, states, masks):

        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = x.view(-1, 32 * 10 * 8)
        x = self.linear1_drop(x)
        x = self.linear1(x)
        x = F.leaky_relu(x)
 
        out = x

        reconstruction, z_mean, z_logvar = None, None, None
        if self.use_vae:
            z_mean = self.linearmean(x)
            z_logvar = self.linearvar(x)
            z = self.reparametrize(z_mean, z_logvar)

            unx = F.relu(self.unlinearlatent(z))
            unx = F.relu(self.unlinear1(unx))
            unx = unx.view(-1, 32, 10, 8)

            unx = self.unconv5(unx)
            unx = self.unconv4(unx)
            unx = self.unconv3(unx)
            unx = self.unconv2(unx)
            reconstruction = self.unconv1(unx)

            out = z

        if hasattr(self, 'gru'):
            if inputs.size(0) == states.size(0):
                x = states = self.gru(x, states * masks)
            else:
                x = x.view(-1, states.size(0), x.size(1))
                masks = masks.view(-1, states.size(0), 1)
                outputs = []
                for i in range(x.size(0)):
                    hx = states = self.gru(x[i], states * masks[i])
                    outputs.append(hx)
                x = torch.cat(outputs, 0)
            out = out + x

        return self.critic_linear(out), out, states, reconstruction, z_mean, z_logvar


    #source https://github.com/coolvision/vae_conv/blob/master/vae_conv_mnist.py
    def compute_vae_loss(self, x, recon_x, mu, logvar):
        #print(recon_x.size(), x.size())
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        MSE = F.l1_loss(x, recon_x)
        return MSE, KLD



def weights_init_mlp(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)


class MLPPolicy(FFPolicy):
    def __init__(self, num_inputs, action_space):
        super(MLPPolicy, self).__init__()

        self.action_space = action_space

        self.a_fc1 = nn.Linear(num_inputs, 64)
        self.a_fc2 = nn.Linear(64, 64)

        self.v_fc1 = nn.Linear(num_inputs, 64)
        self.v_fc2 = nn.Linear(64, 64)
        self.v_fc3 = nn.Linear(64, 1)

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(64, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(64, num_outputs)
        else:
            raise NotImplementedError

        self.train()
        self.reset_parameters()

    @property
    def state_size(self):
        return 1

    def reset_parameters(self):
        self.apply(weights_init_mlp)

        """
        tanh_gain = nn.init.calculate_gain('tanh')
        self.a_fc1.weight.data.mul_(tanh_gain)
        self.a_fc2.weight.data.mul_(tanh_gain)
        self.v_fc1.weight.data.mul_(tanh_gain)
        self.v_fc2.weight.data.mul_(tanh_gain)
        """

        if self.dist.__class__.__name__ == "DiagGaussian":
            self.dist.fc_mean.weight.data.mul_(0.01)

    def forward(self, inputs, states, masks):
        batch_numel = reduce(operator.mul, inputs.size()[1:], 1)
        inputs = inputs.view(-1, batch_numel)

        x = self.v_fc1(inputs)
        x = F.tanh(x)

        x = self.v_fc2(x)
        x = F.tanh(x)

        x = self.v_fc3(x)
        value = x

        x = self.a_fc1(inputs)
        x = F.tanh(x)

        x = self.a_fc2(x)
        x = F.tanh(x)

        return value, x, states
