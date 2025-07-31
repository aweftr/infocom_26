import torch.optim as optim
import torch as th
import torch.nn as nn
from torch.nn.functional import pad
import numpy as np

class QLearner:
    def __init__(self, agent, args):
        self.args      = args
        self.agent     = agent
        self.device    = args.device if th.cuda.is_available() and str(args.device) != 'cpu' else th.device('cpu')
        weight_decay   = getattr(args, 'weight_decay', 0)  # Set to 0 if args.weight_decay is not provided
        self.optimizer = optim.Adam(self.agent.online_net.parameters(), lr=args.lr, weight_decay=weight_decay)
        self.loss_fn   = nn.MSELoss()
        self.learn_cnt = 0

        # Learning rate scheduler
        if args.scheduler == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.99)  # Decay learning rate every 1000 steps
        elif args.scheduler == 'cos':
            self.scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=3000)  # Cosine annealing

    @th.no_grad()
    def update_target_network(self):
        self.agent.update_target_network()
    
    def to_tensor(self, array):
        if isinstance(array, np.ndarray):
            return th.tensor(array, dtype=th.float32)
        elif isinstance(array, th.Tensor):
            return array.clone().detach().float()
        else:
            raise ValueError("Unsupported data type")

    def train(self, memory, batch_size):
        if memory.__len__() < batch_size:
            return None

        # Sample from experience replay buffer
        transitions = memory.sample(batch_size)
        batch = list(zip(*transitions))
        
        # Ensure all data are PyTorch tensors and convert to float32
        max_PMnum = max(max(b.shape[0] for b in batch[0]), max(b.shape[0] for b in batch[4]))
        obs       = th.stack([
                        pad(self.to_tensor(b), (0, 0, 0, 0, 0, max_PMnum - b.shape[0]))
                        if b.shape[0] < max_PMnum else self.to_tensor(b)
                        for b in batch[0]
                    ]).to(self.device)
        feat      = th.stack([self.to_tensor(b) for b in batch[1]]).to(self.device)
        actions   = th.tensor(batch[2], dtype=th.int64).to(self.device)  # actions usually are indices, so int64
        rewards   = th.tensor(batch[3], dtype=th.float32).to(self.device)
        next_obs  = th.stack([
                        pad(self.to_tensor(b), (0, 0, 0, 0, 0, max_PMnum - b.shape[0]))
                        if b.shape[0] < max_PMnum else self.to_tensor(b)
                        for b in batch[4]
                    ]).to(self.device)
        next_feat = th.stack([self.to_tensor(b) for b in batch[5]]).to(self.device)
        dones     = th.tensor(np.array(batch[6], dtype=int)).view(-1).float().to(self.device)  # Convert to float32
        actual_ns = th.tensor(batch[7], dtype=th.float32).to(self.device)  # actual steps

        # Combine into states and next_states
        states      = [obs, feat]
        next_states = [next_obs, next_feat]

        # Compute Q-value of current state and action
        q_values = self.agent.online_net(states)  # (B, 1+2N)
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute next mask
        next_state_mask = th.arange(max_PMnum, device=self.device).expand(len(batch[4]), max_PMnum)  # shape (B, N)
        next_lengths = th.tensor([b.shape[0] for b in batch[4]], device=self.device)           # shape (B) 每个样本真实 PM 数
        next_state_mask = (next_state_mask < next_lengths.unsqueeze(1))  # shape (B, N)，bool
        # next q mask
        q_mask = th.zeros((next_state_mask.shape[0], 1+2*max_PMnum), dtype=th.bool, device=self.device)
        q_mask[:, 0] = True
        q_mask[:, 1::2] = next_state_mask
        q_mask[:, 2::2] = next_state_mask
        
        # Compute Q-value of next state (from online network)
        next_q_online = self.agent.online_net(next_states)              # (B, 1+2N)
        next_q_online_masked = next_q_online.masked_fill(~q_mask, -1e9)
        next_actions = next_q_online_masked.argmax(dim=1, keepdim=True) # (B,1)

        # target_net 评估 Q(s', a*)
        with th.no_grad():
            next_q_target = self.agent.target_net(next_states)          # (B, A_max)
            next_q_value  = next_q_target.gather(1, next_actions).squeeze(1)

        # Compute expected Q-value (multi-step return)
        expected_q_value = rewards + (self.args.gamma ** actual_ns) * next_q_value * (1 - dones)

        # Backpropagation
        loss = self.loss_fn(q_value, expected_q_value)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.learn_cnt += 1
        if self.learn_cnt % self.args.target_update_interval == 0:
            self.update_target_network()
        
        self.scheduler.step()

        return loss.item()
