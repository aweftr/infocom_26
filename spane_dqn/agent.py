import torch.nn as nn
import torch.nn.init as init
import torch as th
import numpy as np
import os

'''
obs: tensor([
[[40., 90.],[40., 90.]],
[[40., 90.],[40., 90.]],
[[40., 90.],[40., 90.]],
[[40., 90.],[40., 90.]],
[[40., 90.],[40., 90.]]
]) 

feat: tensor(
[   [[2., 4.],[0., 0.]],
    [[0., 0.],[2., 4.]] ]
)
'''

def init_weights(m):
    if isinstance(m, nn.Linear):
        init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            init.constant_(m.bias, 0)

def check_obs(obs):  # Ensure the type is tensor and check the shape
    if not isinstance(obs, th.Tensor): obs = th.tensor(obs, dtype=th.float32)
    if   len(obs.shape) == 3:  return obs.unsqueeze(0)
    elif len(obs.shape) == 4:  return obs       # (batch size, ., ., .)
    else                    :  raise ValueError(f"Shape Incorrect, obs.shape: {obs.shape}")

def check_feat(feat):  # Ensure the type is tensor and check the shape
    if not isinstance(feat, th.Tensor): feat = th.tensor(feat, dtype=th.float32)
    if   len(feat.shape) == 1:  return feat.unsqueeze(0)
    elif len(feat.shape) == 2:  return feat       # (batch size, ...)
    else                     :  raise ValueError(f"Shape Incorrect, feat.shape: {feat.shape}")

def normalize_data(obs, feat, cpu, mem):
    obs[:, :, :, 0]  = obs[:, :, :, 0] / cpu
    obs[:, :, :, 1]  = obs[:, :, :, 1] / mem
    feat[:, 0] = feat[:, 0] / cpu
    feat[:, 1] = feat[:, 1] / mem
    return obs, feat

def get_layers(relu_no, initial_width, width, final_width):
    layers = nn.ModuleList([nn.Linear(initial_width, width)])
    for _ in range(relu_no-1):
        layers.extend([
            nn.ReLU(),
            nn.Linear(width, width)
        ])
    layers.extend([
        nn.ReLU(),
        nn.Linear(width, final_width)
    ])
    return layers

def _param_device(module: nn.Module):
    return next(module.parameters()).device

class DQNAgent_sym(nn.Module): 
    def __init__(self, cpu, mem, nn_widths, aggs, easy):
        super(DQNAgent_sym, self).__init__()
        self.cpu = cpu
        self.mem = mem
        self.aggs = aggs  # aggregation (mean)
        self.easy = easy
        feat_dim = 3

        # embedding layers
        width_server, relu_no = nn_widths['server']
        self.server_embedding = get_layers(relu_no, 4, width_server, width_server)
        width_cluster = width_server * len(aggs)
        # Value layers
        width_value, relu_no = nn_widths['value']
        self.value_layers     = get_layers(relu_no, width_cluster + feat_dim, width_value, 1)
        # Advantages PM layers
        width_adv, relu_no = nn_widths['adv']
        self.adv_layers = get_layers(relu_no, width_server + width_cluster + feat_dim, width_adv, 2)
        # Advantages new PM layers
        self.adv_newPM_layers = get_layers(relu_no, width_cluster + feat_dim, width_adv, 1)

        self.apply(init_weights)  # initialize weights

    def forward(self, state, is_easy_valid=False):
        # Data process
        dev = _param_device(self)
        obs, feat = state[0], state[1]
        obs, feat = check_obs(obs).to(dev), check_feat(feat).to(dev)   # Ensure obs is a tensor with shape (B, N, 2, 2) (default B = 1), and feat has shape (B, 2, 2, 2).
        obs, feat = normalize_data(obs, feat, self.cpu, self.mem)
        B = obs.shape[0]  # batch size
        N = obs.shape[1]  # max PM num
        
        # padding mask
        with th.no_grad():
            pm_mask = (obs.abs().sum(dim=(2, 3)) > 0)  # shape: (B, N)
            empty_mask = ~pm_mask.any(dim=1)
            if empty_mask.any():
                pm_mask[empty_mask, 0] = True  # 保留至少一个 PM
            pm_mask = pm_mask.to(dev)

        if not is_easy_valid:
            # calc server
            server_embeddings = obs.view(B, N, 4)
            for layer in self.server_embedding:
                server_embeddings = layer(server_embeddings)
            # calc cluster embedding
            f_mean = lambda x, mask: (x * mask.unsqueeze(2)).sum(dim=1) / mask.sum(dim=1, keepdim=True)
            cluster_embedding = f_mean(server_embeddings, pm_mask)

            # calc value
            value = th.cat([cluster_embedding, feat], dim=1)
            for layer in self.value_layers:
                value = layer(value)

            # calc advantages
            cluster_embedding_rpt = cluster_embedding.unsqueeze(1).repeat(1, N, 1)
            feat_rpt = feat.unsqueeze(1).repeat(1, N, 1)
            advs = th.cat([server_embeddings, cluster_embedding_rpt, feat_rpt], dim=2)
            for layer in self.adv_layers:
                advs = layer(advs)
            advs = advs.view(B, 2 * N)
            # calc new PM adv
            newPM_advs = th.cat([cluster_embedding, feat], dim=1)
            for layer in self.adv_newPM_layers:
                newPM_advs = layer(newPM_advs)
            final_advs = th.cat([newPM_advs, advs], dim=1)  # (B, 1 + 2*N)
        
            f_adv_mean = lambda advs, mask: (advs * mask).sum(dim=1, keepdim=True) / mask.sum(dim=1, keepdim=True)
            adv_mask = th.zeros((pm_mask.shape[0], 1+2*pm_mask.shape[1]), dtype=th.bool, device=dev)
            adv_mask[:, 0] = True
            adv_mask[:, 1::2] = pm_mask
            adv_mask[:, 2::2] = pm_mask
            q_values = value + final_advs - f_adv_mean(final_advs, adv_mask)
            return q_values

        else:
            raise RuntimeError("Didn't consider!")
            advs = obs.view(B, N, 4)  
            for layer in self.adv_layers:
                advs = layer(advs)
            advs = advs.view(B, 2 * N)   
            
            newPM_advs = th.cat([cluster_embedding_rpt, feat_rpt], dim=1)
            for layer in self.adv_newPM_layers:
                newPM_advs = layer(newPM_advs)
                
            final_advs = th.cat([newPM_advs, advs], dim=1)
            return final_advs
        
class DoubleDQNAgent:  # Split the DQN network into two parts to improve stability. See Double DQN for details.
    def __init__(self, args, mode='basic'):
        if mode == 'sym':
            self.online_net = DQNAgent_sym(args.cpu, args.mem, args.nn_width.copy(), args.cluster_agg, easy=0)
            self.target_net = DQNAgent_sym(args.cpu, args.mem, args.nn_width.copy(), args.cluster_agg, easy=0)
        else:
            raise ValueError("DoubleDQNAgent mode incorrect")
        self.update_target_network()

    def update_target_network(self):
        self.target_net.load_state_dict(self.online_net.state_dict())
    
    def select_action(self, state, epsilon, is_easy_valid=False):  # state = {'obs':..., 'feat':..., 'avail':...}
        avail = state['avail']

        if len(avail.shape) == 1:  # Single env case
            if np.random.rand() < epsilon:
                valid_action = np.random.choice(np.where(avail == 1)[0])
                return valid_action.item()
            else:
                with th.no_grad():
                    q_values = self.online_net([state['obs'], state['feat']])
                    avails = np.expand_dims(avail, axis=0)
                    q_values[avails == 0] = float('-inf')  # Set the Q-values of unavailable actions to negative infinity
                    return th.argmax(q_values, dim=1).cpu().item()
                
        elif len(avail.shape) == 2: # 多个env
            print(f"more env path")
            raise ValueError()
            if np.random.rand() < epsilon:
                valid_actions = [np.random.choice(np.where(avail[i] == 1)[0]) for i in range(state[0].shape[0])]
                return np.array(valid_actions)
            else:
                with th.no_grad():
                    q_values = self.online_net([state['obs'], state['feat']])
                    q_values[avail == 0] = float('-inf')  # Set the Q-values of unavailable actions to negative infinity
                    return th.argmax(q_values, dim=1).cpu().numpy()
                    
        else:
            raise ValueError("avail shape incorrect!")
           
    def save(self, filepath):
        dir_path = os.path.dirname(filepath)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        th.save(self.online_net.state_dict(), filepath)

    def load(self, filepath):
        self.online_net.load_state_dict(th.load(filepath, weights_only=True, map_location='cpu'))
        self.update_target_network()
