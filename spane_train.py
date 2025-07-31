import os
import torch
import numpy as np
import pandas as pd
from types import SimpleNamespace
import argparse
import random
from tqdm import trange
from schedgym.sched_env_minusage_spane import SchedEnv
from spane_dqn.agent import DoubleDQNAgent
from spane_dqn.learner import QLearner
from spane_dqn.replay_memory import ReplayMemory
from common import linear_decay, trimmed_mean

# 固定随机种子
random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
torch.backends.cudnn.deterministic = True

# 一些超参数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = 'data/Huawei-East-1-lt.csv'
valid_inds = np.load('data/valid_random_time_150.npy')
logpath = '/root/infocom_26/runs_spanedqn'
env_config = {
    'env': 'recovering',
    'N': 100,  
    'cpu': 40,
    'mem': 90,
    'allow_release': True, 
    'double_thr': 10,       # Memory threshold, if VM memory >= 10, the VM equally distributed
}
es_tol = 1e-5

# 获取 data
def getData(path, double_thr=1e10):
    csv_data = pd.read_csv(path)
    data = csv_data.to_dict('records')
    for item in data:
        item['at'] = int(item['at'])
        item['lt'] = int(item['lt'])
        item['is_double'] = int(item['mem'] > double_thr)
    return data

def main(sp: argparse.Namespace):
    '1. Hyperparameters'
    args = SimpleNamespace(**env_config)  # Initialize configuration class
    args.alg = 'dqn'
    args.mode = 'sym'
    args.reward_type = 'basic'
    args.N_vm = 1000
    args.device = device

    if args.mode == 'basic':
        args.nn_width = [sp.width] * sp.layer_no   # Neural network hidden layer dimensions
    if args.mode == 'sym':
        args.nn_width = {
            'server': (sp.nn_width_server, sp.nn_num_server),
            'value': (sp.nn_width_value, sp.nn_num_value),
            'adv': (sp.nn_width_adv, sp.nn_num_adv)
        }
        aggs_dicts = {0: 'mean', 1: 'max', 2: 'std'}
        args.cluster_agg = [aggs_dicts[i] for i in sp.cluster_agg]

    # DQN-specific parameters
    args.eps_1 = 0.6  # At the beginning of online training, the probability of the agent selecting an action randomly
    args.eps_2 = 0.0  # At the end of online training, the probability of the agent selecting an action randomly
    args.n_step = 50  # Multi-step return for the loss function (larger n_step reduces bias but increases variance)
    args.memory_capacity = 100000  # Size of the replay buffer
    args.weight_decay = 1e-8  # L2 regularization weight
    args.lr = sp.lr # Initial learning rate
    args.scheduler = 'step'  # Type of learning rate scheduler, see learner for details
    args.gamma = 0.99  # Discount factor for the n-step reward

    # Epoch parameters
    args.online_epoch = 10000 # 10000 # Number of batches of data learned in each epoch
    args.run_interval = 5
    args.run_no = 10  # Number of data collections with the environment in each epoch
    args.batch_size = 1024
    args.valid_interval = 250  #250 # Interval (in epochs) for validation
    args.valid_num = 150 #150 # Number of experiments during validation, average taken
    args.target_update_interval = 100 # 100

    '2. Functions'
    def push_memory(replay_memory, obs, feat, action, reward, next_obs, next_feat, done):
        replay_memory.push(obs, feat, action, reward, next_obs, next_feat, done)
            
    def run(env: SchedEnv, initial_index, agent: DoubleDQNAgent, memory: ReplayMemory, eps, is_valid):
        ''' # Interact with the environment to return rewards or store experiences (required for online training) '''
        # Reset environment
        state = env.reset(initial_index, N_vm=args.N_vm)

        tot_reward = 0
        done = False
        while not done:
            # Agent action selection
            action = agent.select_action(state, eps)
            state2, reward, done = env.step(action)
            tot_reward += reward

            # Store experience in memory if training
            if not is_valid:
                obs = state['obs']
                feat = state['feat']
                next_obs = state2['obs']
                next_feat = state2['feat']
                push_memory(memory, obs, feat, action, reward, next_obs, next_feat, done)
            state = state2

        return tot_reward

    def validate(env, agent, args, epoch):
        pm_usage_arr = []
        total_pm_nums = []

        for i in trange(args.valid_num):
            initial_index = valid_inds[i]
            run(env, initial_index, agent, None, 0, True)
            max_pm_num = env.maximum_pm_num
            pm_usage = env.total_pm_usage
            pm_usage_arr.append(pm_usage)
            total_pm_nums.append(max_pm_num)

        # Save model
        model_path = f'{logpath}/{epoch}.th'
        model_save(agent, model_path)

        pm_usage_mean = sum(pm_usage_arr) / len(pm_usage_arr)
        pm_usage_tmean = trimmed_mean(pm_usage_arr)
        pm_num_mean = trimmed_mean(total_pm_nums)
        return pm_usage_mean, pm_usage_tmean, pm_num_mean

    def model_save(agent, model_path):
        dir_path = os.path.dirname(model_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        agent.save(model_path)

    def interact(env, agent, memory_on, eps):
        initial_index = np.random.randint(0, 100000)
        return run(env, initial_index, agent, memory_on, eps, False)

    '3. Preparation'
    env_data = getData(DATA_PATH, args.double_thr)
    env = SchedEnv(args.cpu, args.mem, env_data, args.double_thr, args.reward_type)
    agent = DoubleDQNAgent(args, args.mode)
    agent.online_net.to(device)
    agent.target_net.to(device)
    memory_on = ReplayMemory(args.memory_capacity, args.n_step, args.gamma)
    learner = QLearner(agent, args)

    '4. Pre-collect data'
    precollect_epoch = 20  # 20
    for _ in trange(precollect_epoch):
        initial_time = np.random.randint(0, 100000)
        eps = args.eps_1
        run(env, initial_time, agent, memory_on, eps, False)

    '5. Online training'
    epoch = 0
    best_pm_num = 100000000
    best_result = {}
    best_result_epoch = 0
    same_result_count = 0
    last_pm_num = 0
    for _ in trange(args.online_epoch + 1):
        # Collect data through interaction
        eps = linear_decay(epoch, args.online_epoch, args.eps_1, args.eps_2)
        if epoch % args.run_interval == 0:
            for _ in range(args.run_no):
                interact(env, agent, memory_on, eps)
                
        # Model learning (train on data sampled from memory)
        loss = learner.train(memory_on, args.batch_size)
        if loss == None: loss = 0
        
        # Validation phase
        if epoch % args.valid_interval == 0:
            pm_usage_mean, pm_usage_tmean, max_pm_num = validate(env, agent, args, epoch)
            if max_pm_num < best_pm_num:
                best_pm_num = max_pm_num
                best_result['pm_usage_mean'] = pm_usage_mean
                best_result['pm_usage_tmean'] = pm_usage_tmean
                best_result_epoch = epoch
            same_result_count = same_result_count + 1 if abs(max_pm_num - last_pm_num) < es_tol else 0
            last_pm_num = max_pm_num
            # Early stopping if the result is the same for three consecutive validations
            if same_result_count >= 3:
                break
        epoch += 1

    return best_pm_num, best_result, best_result_epoch


if __name__ == "__main__":
    # Parameters
    sp_dict = {
        'nn_width_server': 8,    
        'nn_num_server': 1,     
        'nn_width_value': 8,
        'nn_num_value': 1,
        'nn_width_adv': 16,
        'nn_num_adv': 1,    
        'cluster_agg': [0],  
        'lr': 0.01     
    }
    sp = SimpleNamespace(**sp_dict)
    
    # Run
    best_pm_num, best_result, best_result_epoch = main(sp)
    print(f"max pm mean: {best_pm_num}, pm usage mean: {best_result['pm_usage_mean']}\n")
    print(f"pm usage trimmed mean: {best_result['pm_usage_tmean']}, best epoch: {best_result_epoch}")

    # Store results
    with open(f'spane_dqn_result.txt', 'a') as f:
        f.write(f"max pm mean: {best_pm_num}, pm usage mean: {best_result['pm_usage_mean']}\n")
        f.write(f"pm usage trimmed mean: {best_result['pm_usage_tmean']}, best epoch: {best_result_epoch}\n")
