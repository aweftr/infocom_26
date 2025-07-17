''' Evaluate the results of First Fit and Balance Fit '''

import numpy as np
from schedgym.sched_env import SchedEnv 
from baseline_agent import get_fit_func
from tqdm import trange
from common import trimmed_mean
from multiprocessing import Pool
from ppo_agent import Agent, flatten_observation
import os
import torch
DATA_PATH = "data/Huawei-East-1-lt.csv"

# A. Validation Result
# valid_inds = np.load('data/valid_random_time_150.npy')
# num_episodes = 150
# B. Test Result
valid_inds = np.load('data/test_random_time_1000.npy')
valid_Nvms = np.load("data/test_random_Nvm.npy")
num_episodes = 1000
use_gpu = False
device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")

first_fit = get_fit_func(0, 40, 90)
def run_episode(env: SchedEnv, agent, index, N_vm):
    state = env.reset(index, N_vm=N_vm)
    done = False
    while not done:
        with torch.no_grad():
            state_flatten = torch.Tensor(flatten_observation(state)).to(device)
            avail = torch.tensor(state["avail"], dtype=torch.bool).to(device)
            action = agent.get_action_inference(state_flatten, avail=avail)
            # breakpoint()
            state, _, done = env.step(action)

    return env.get_attr('total_wait_time')

def main():
    # Environment parameters
    N = 5  # Number of servers
    cpu = 40  # Total CPU per NUMA
    mem = 90  # Total memory per NUMA

    env = SchedEnv(N, cpu, mem, DATA_PATH, 10)
    agent = Agent(env)
    # path = os.listdir("runs")[-3]
    # path = "ppo_train__1__1752493459" # vanilla train 9w+
    path = "ppo_BCpretrain__1__1752562728" # BC train
    print(path)
    agent.load_state_dict(torch.load(os.path.join("runs", path, "model.pth")))
    agent.to(device)
    # breakpoint()
    # for i in os.listdir("runs")
    first_fit = get_fit_func(0, cpu, mem)  # First Fit agent
    bal_fit = get_fit_func(2, cpu, mem)    # Balance Fit agent

    # Run multiple episodes
    res_f = []  # Results for First Fit
    res_b = []  # Results for Balance Fit
    # res_ppo = []
    args = [(env, agent, valid_inds[episode], valid_Nvms[episode]) for episode in range(num_episodes)]
    with Pool(processes=8) as pool:
        res_ppo = pool.starmap(run_episode, args)
    # for episode in trange(num_episodes):
    #     index = valid_inds[episode]  # Select a valid test index
    #     # res_f.append(run_episode(env, first_fit, index))
    #     # res_b.append(run_episode(env, bal_fit, index))
    #     res_ppo.append(run_episode(env, agent, index))
    # print(f"first fit: {trimmed_mean(res_f)}")
    # print(f"first fit: {np.mean(res_f)}")
    # print(f"balance fit: {trimmed_mean(res_b)}")
    # print(f"balance fit: {np.mean(res_b)}")
    print(f"ppo: {trimmed_mean(res_ppo)}")
    print(f"ppo: {np.mean(res_ppo)}")
    '''
    first fit: 89822.5225
    first fit: 362634.624
    balance fit: 86009.90875
    balance fit: 336486.767
    '''

if __name__ == "__main__":
    main()
