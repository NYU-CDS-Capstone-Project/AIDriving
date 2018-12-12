import copy
import glob
import os
import time

import torch
import numpy as np

class ModelLogger(object):
    def __init__(self, log_dir, num_processes):
        try:
            os.makedirs(log_dir)
        except OSError:
            files = glob.glob(os.path.join(log_dir, '*.monitor.csv'))
            for f in files:
                os.remove(f)

        # These variables are used to compute average rewards for all processes.
        self.total_episode_rewards_avg = []
        self.total_episode_lengths_avg = []
        self.total_value_loss = []
        self.total_action_loss = []
        self.total_recon_loss = []
        self.total_kld_loss = []
        self.total_entropy = []


        self.episode_rewards = torch.zeros([num_processes, 1])
        self.final_rewards = torch.zeros([num_processes, 1])
        self.episode_lengths = torch.zeros([num_processes, 1])
        self.final_lengths = torch.zeros([num_processes, 1])

        self.reward_avg = 0
        self.length_avg = 0

    def update_reward(self, reward, masks):
        self.episode_rewards += reward
        self.episode_lengths += 1
        # If done then clean the history of observations.
        self.final_rewards *= masks
        self.final_lengths *= masks
        self.final_rewards += (1 - masks) * self.episode_rewards
        self.final_lengths += (1 - masks) * self.episode_lengths
        self.episode_rewards *= masks
        self.episode_lengths *= masks

    def save_model(self, save_dir, model, envs, algo, env_name, name, cuda):
        save_path = os.path.join(save_dir, algo)
        try:
            os.makedirs(save_path)
        except OSError:
            pass

        save_model = model
        if cuda:
            save_model = copy.deepcopy(model).cpu()

        save_model = [save_model,
                        hasattr(envs, 'ob_rms') and envs.ob_rms or None]

        torch.save(save_model, os.path.join(save_path, env_name + "_" + name + ".pt"))
        np.save(os.path.join(save_path, env_name + "_" + name + ".npy"), 
            np.asarray([self.total_episode_rewards_avg, 
                        self.total_episode_lengths_avg, 
                        self.total_value_loss, 
                        self.total_action_loss, 
                        self.total_entropy,
                        self.total_recon_loss,
                        self.total_kld_loss]))


    def print_log(self, value_loss, action_loss, dist_entropy, recon_loss, kld_loss, num_processes, num_steps, nthupdate, start):
        self.reward_avg = 0.99 * self.reward_avg + 0.01 * self.final_rewards.mean()
        self.length_avg = 0.99 * self.length_avg + 0.01 * self.final_lengths.mean()
        self.total_episode_rewards_avg.append(self.reward_avg)
        self.total_episode_lengths_avg.append(self.length_avg)
        self.total_value_loss.append(value_loss)
        self.total_action_loss.append(action_loss)
        self.total_entropy.append(dist_entropy)
        self.total_recon_loss.append(recon_loss)
        self.total_kld_loss.append(kld_loss)

        end = time.time()
        total_num_steps = (nthupdate + 1) * num_processes * num_steps

        print(
            "Updates {}, num timesteps {}, FPS {}, running avg reward {:.3f}, running avg eplen {:2f}, entropy {:.5f}, value loss {:.5f}, policy loss {:.5f}, recon loss {:.5f}, kld loss {:.5f}".
            format(
                nthupdate,
                total_num_steps,
                int(total_num_steps / (end - start)),
                self.reward_avg,
                self.length_avg,
                dist_entropy,
                value_loss,
                action_loss,
                recon_loss,
                kld_loss
            )
        )
