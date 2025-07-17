import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import numpy as np
from typing import Callable

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

def flatten_observation(observation):
    return np.concatenate((observation["obs"].reshape(-1), observation["feat"]))

class Agent(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(env.observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(env.observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, env.action_space.n), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None, avail=None):
        logits = self.actor(x)
        logits = logits.masked_fill(~avail, -1e10)
        # print(logits)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)
    
    def get_action_inference(self, x, avail=None):
        logits = self.actor(x)
        logits = logits.masked_fill(~avail, -1e10)
        action = torch.argmax(logits)
        return action


class MyVectorEnv:
    def __init__(self, make_env: Callable, num_envs, exceed_vm):
        self.num_envs = num_envs
        self.envs = [make_env() for _ in range(num_envs)]
        self.exceed_vm = exceed_vm
        
        self.single_observation_space = self.envs[0].observation_space
        self.single_action_space = self.envs[0].action_space

    def reset(self, seed=None, options=None):
        obs = []
        avail = []
        rseed = seed
        for i, env in enumerate(self.envs):
            observation = env.reset(exceed_vm=self.exceed_vm, seed=rseed, options=options)
            obs.append(flatten_observation(observation))
            avail.append(observation["avail"])
            rseed += 1
        return np.array(obs), np.array(avail)

    def step(self, actions):
        obs, rewards, truncated, avails, infos = [], [], [], [], []
        for env, action in zip(self.envs, actions):
            info = {}
            o, r, done = env.step(action) 
            if done:
                info["done_info"] = o
                info["total_wait_time"] = env.total_wait_time
                o = env.reset(exceed_vm=self.exceed_vm)
            obs.append(flatten_observation(o))
            rewards.append(r)
            truncated.append(done)
            avails.append(o["avail"])
            infos.append(info)
        return (
            np.array(obs),
            np.array(rewards),
            np.array(truncated),
            np.array(avails),
            infos
        )
    
    def sample_action(self, avails):
        sampled_action = []
        for i, row in enumerate(avails):
            row_true_indices = np.where(row)[0]
            if len(row_true_indices) > 0:
                sample = np.random.choice(row_true_indices)
                sampled_action.append(sample)
        sampled_action = np.array(sampled_action)
        return sampled_action
    
    def close(self):
        for env in self.envs:
            env.close()