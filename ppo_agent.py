# %%
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
    # return [VMfeatues, PMfeatures]
    return [observation["feat"], observation["obs"].reshape(-1)]
    # return np.concatenate((observation["obs"].reshape(-1), observation["feat"]))

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


# %%
class TransformerPPOAgent(nn.Module):
    def __init__(self, vm_dim, pm_dim, d_model=128, nhead=4, num_layers=2, max_pms=200):
        super().__init__()
        self.d_model = d_model
        self.max_pms = max_pms

        # Token type embeddings: 0 = VM, 1 = PM
        self.token_type_embedding = nn.Embedding(2, d_model)

        # Feature embeddings
        self.vm_embedding = layer_init(nn.Linear(vm_dim, d_model))
        self.pm_embedding = layer_init(nn.Linear(pm_dim, d_model))

        # Positional embeddings (optional, per role)
        self.vm_positional = nn.Parameter(torch.zeros(1, 1, d_model))  # for VM
        self.pm_positional = nn.Parameter(torch.zeros(1, max_pms, d_model))  # for PMs

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Policy head: outputs one logit per token (1 + N actions)
        self.policy_head = layer_init(nn.Linear(d_model, 1))

        # Value head: use VM token (index 0)
        self.value_head = layer_init(nn.Linear(d_model, 1))

    def forward(self, vm_features, pm_features, pm_mask):
        """
        vm_features: [B, vm_dim]
        pm_features: [B, N, pm_dim], N is the number of NUMAs
        pm_mask: [B, 1 + N] with 1 = valid PM, 0 = padding, the first dimension is VM and is always 1
        """
        B, N, _ = pm_features.shape

        # Embed VM token
        vm_emb = self.vm_embedding(vm_features).unsqueeze(1)  # [B, 1, d_model]
        vm_emb += self.token_type_embedding(torch.zeros(B, 1, dtype=torch.long, device=vm_features.device))
        vm_emb += self.vm_positional  # learned positional embedding

        # Embed PM tokens
        pm_emb = self.pm_embedding(pm_features)  # [B, N, d_model]
        pm_emb += self.token_type_embedding(torch.ones(B, N, dtype=torch.long, device=vm_features.device))
        pm_emb += self.pm_positional[:, :N, :]  # learned PM position (optional)

        # Combine tokens: [VM] + [PM_1, ..., PM_N]
        tokens = torch.cat([vm_emb, pm_emb], dim=1)  # [B, 1 + N, d_model]

        # Build attention mask: True = ignore
        # attn_mask = torch.cat([torch.ones(B, 1, device=pm_mask.device), pm_mask], dim=1) == 0

        # Encode with Transformer
        encoded = self.transformer(tokens, src_key_padding_mask=~pm_mask)  # [B, 1 + N, d_model]
        # encoded = self.transformer(tokens, src_key_padding_mask=(pm_mask == 0))  # [B, 1 + N, d_model]

        return encoded

    def get_action_and_value(self, vm_features, pm_features, pm_mask, action_mask, action=None):
        encoded = self.forward(vm_features, pm_features, pm_mask)

        # Policy logits for each token (action 0 = open new PM)
        logits = self.policy_head(encoded).squeeze(-1)  # [B, 1 + N]
        # breakpoint()
        logits = logits.masked_fill(~action_mask, -1e10)
        # logits = logits.masked_fill((action_mask == 0), -1e10)
        # print(logits.shape, action)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()

        # Value from VM token
        # TODO can use mean(dim=1) to average over all VM + PMs. 
        value = self.value_head(encoded[:, 0])  # [B, 1]
        return action, probs.log_prob(action), probs.entropy(), value

    def get_action(self, vm_features, pm_features, pm_mask, action_mask):
        encoded = self.forward(vm_features, pm_features, pm_mask)

        # Policy logits for each token (action 0 = open new PM)
        logits = self.policy_head(encoded).squeeze(-1)  # [B, 1 + N]
        logits = logits.masked_fill(~action_mask, -1e10)
        # logits = logits.masked_fill((action_mask == 0), -1e10)
        action = torch.argmax(logits, dim=1)
        return action
    
    def get_value(self, vm_features, pm_features, pm_mask):
        encoded = self.forward(vm_features, pm_features, pm_mask)
        value = self.value_head(encoded[:, 0])
        return value

# # %%
# a = TransformerPPOAgent(3, 2)
# VM_features = torch.randn((2, 3))

# from torch.nn.utils.rnn import pad_sequence
# PM_features1 = torch.randn((6, 2))
# PM_features2 = torch.randn((12, 2))
# pm_features_list = [PM_features1, PM_features2]
# pm_padded = pad_sequence(pm_features_list, batch_first=True)

# pm_mask = torch.tensor([
#     [1] * (pm.shape[0] + 1) + [0] * (pm_padded.shape[1] - pm.shape[0])
#     for pm in pm_features_list
# ], dtype=torch.bool)  # [B, max_pms]
# # %%
# a.get_action_and_value(VM_features, pm_padded, pm_mask, pm_mask)

# %%
class MyVectorEnv:
    def __init__(self, make_env: Callable, num_envs, N_vm):
        self.num_envs = num_envs
        self.envs = [make_env() for _ in range(num_envs)]
        self.N_vm = N_vm
        

    def reset(self, seed=None, options=None):
        obs = []
        avail = []
        rseed = seed
        for i, env in enumerate(self.envs):
            observation = env.reset(N_vm=self.N_vm, seed=rseed, options=options)
            obs.append(flatten_observation(observation))
            avail.append(observation["avail"])
            rseed += 1
        return obs, avail

    def step(self, actions):
        obs, rewards, truncated, avails, infos = [], [], [], [], []
        for env, action in zip(self.envs, actions):
            info = {}
            o, r, done = env.step(action) 
            if done:
                info["done_info"] = o
                info["total_pm_usage"] = env.total_pm_usage
                o = env.reset(N_vm=self.N_vm)
            obs.append(flatten_observation(o))
            rewards.append(r)
            truncated.append(done)
            avails.append(o["avail"])
            infos.append(info)
        return (
            obs,
            np.array(rewards),
            np.array(truncated),
            avails,
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

class MyVectorEnvWithIndex:
    def __init__(self, make_env: Callable, num_envs, N_vm):
        self.num_envs = num_envs
        self.envs = [make_env() for _ in range(num_envs)]
        self.N_vm = N_vm
        

    def reset(self, indices, seed=None, options=None):
        obs = []
        avail = []
        for i, env in enumerate(self.envs):
            observation = env.reset(N_vm=self.N_vm, index=indices[i])
            obs.append(flatten_observation(observation))
            avail.append(observation["avail"])
        return obs, avail

    def step(self, actions):
        obs, rewards, truncated, avails, infos = [], [], [], [], []
        for env, action in zip(self.envs, actions):
            info = {}
            o, r, done = env.step(action) 
            if done:
                info["done_info"] = o
                info["total_pm_usage"] = env.total_pm_usage
            obs.append(flatten_observation(o))
            rewards.append(r)
            truncated.append(done)
            avails.append(o["avail"])
            infos.append(info)
        return (
            obs,
            np.array(rewards),
            np.array(truncated),
            avails,
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

# from schedgym.sched_env_minusage import SchedEnv
# def make_env():
#     return SchedEnv(40, 90, "data/Huawei-East-1-lt.csv", 10, random_reset=True)

# envs = MyVectorEnv(make_env, 4, 1000)
# obs, avails = envs.reset(seed=1)
# obs, rewards, truncated, avails, infos = envs.step(envs.sample_action(avails))
# envs.close()