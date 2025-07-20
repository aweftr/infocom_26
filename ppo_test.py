''' Evaluate the results of First Fit and Balance Fit '''

import numpy as np
from schedgym.sched_env_minusage import SchedEnv 
from baseline_agent import get_fit_func
from tqdm import trange
from common import trimmed_mean
from multiprocessing import Pool
from ppo_agent import *
import os
import torch
DATA_PATH = "data/Huawei-East-1-lt.csv"

# A. Validation Result
# valid_inds = np.load('data/valid_random_time_150.npy')
# num_episodes = 150
# B. Test Result
valid_inds = np.load('data/test_random_time_1000.npy')
num_episodes = 1000
N_vm = 1000
use_gpu = True
device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")

first_fit = get_fit_func(0, 40, 90)
def run_episode(env: SchedEnv, agent, index):
    state = env.reset(index, N_vm=N_vm)
    done = False
    while not done:
        with torch.no_grad():
            obs = flatten_observation(state)
            vm_features = torch.Tensor(obs[0]).reshape(1, -1).to(device)
            pm_features = torch.Tensor(obs[1]).reshape(1, -1, 2).to(device)
            pm_mask = torch.ones(pm_features.shape[1] + 1, dtype=torch.bool).reshape(1, -1).to(device)
            action_mask = torch.tensor(state["avail"], dtype=torch.bool).reshape(1, -1).to(device)
            action = agent.get_action(vm_features, pm_features, pm_mask, action_mask)
            # action = agent(state)
            state, _, done = env.step(action)

            # action = agent.get_action_inference(state_flatten, avail=avail)

    return env.total_pm_usage, env.maximum_pm_num

def main():
    # Environment parameters
    cpu = 40  # Total CPU per NUMA
    mem = 90  # Total memory per NUMA

    env = SchedEnv(cpu, mem, DATA_PATH, 10)
    agent = TransformerPPOAgent(3, 2)
    # path = os.listdir("runs")[-3]
    # path = "ppo_train__1__1752493459" # vanilla train 9w+
    path = "ppo_BCpretrain__1__1752925552" # BC train
    print(path)
    agent.load_state_dict(torch.load(os.path.join("runs", path, "model.pth")))
    agent.to(device)
    # a = run_episode(env, agent, valid_inds[0], 1000)
    # breakpoint()
    # for i in os.listdir("runs")
    # Run multiple episodes

    res_ppo = []
    # first_fit = get_fit_func(0, cpu, mem)  # First Fit agent
    # args = [(env, agent, valid_inds[episode]) for episode in range(num_episodes)]
    # with Pool(processes=8) as pool:
    #     result = pool.starmap(run_episode, args)
    for episode in trange(num_episodes):
        index = valid_inds[episode]  # Select a valid test index
    #     # res_f.append(run_episode(env, first_fit, index))
    #     # res_b.append(run_episode(env, bal_fit, index))
        res_ppo.append(run_episode(env, agent, index))
    # print(f"first fit: {trimmed_mean(res_f)}")
    # print(f"first fit: {np.mean(res_f)}")
    # print(f"balance fit: {trimmed_mean(res_b)}")
    # print(f"balance fit: {np.mean(res_b)}")
    result = np.array(res_ppo)
    res = result[:, 0]
    maxi = result[:, 1]
    print(f"ppo pm usage trimmed mean: {trimmed_mean(res)}")
    print(f"ppo pm usage mean: {np.mean(res)}")
    print(f"ppo max pm mean: {np.mean(maxi)}")
    '''
    first fit: 89822.5225
    first fit: 362634.624
    balance fit: 86009.90875
    balance fit: 336486.767
    '''

if __name__ == "__main__":
    main()
