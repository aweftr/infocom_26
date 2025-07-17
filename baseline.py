''' Evaluate the results of First Fit and Balance Fit '''

import numpy as np
from schedgym.sched_env import SchedEnv 
from baseline_agent import get_fit_func
from tqdm import trange
from multiprocessing import Pool
import copy
from common import trimmed_mean
DATA_PATH = "data/Huawei-East-1-lt.csv"

# A. Validation Result
# valid_inds = np.load('data/valid_random_time_150.npy')
# valid_Nvms = np.load("data/valid_random_Nvm.npy")
# num_episodes = 150
# B. Test Result
valid_inds = np.load('data/test_random_time_1000.npy')
valid_Nvms = np.load("data/test_random_Nvm.npy")
num_episodes = 1000

first_fit = get_fit_func(0, 40, 90)
def run_episode(env: SchedEnv, agent, index, N_vm):
    # 1. Use First Fit to determine N_vm for the comparison experiment

    # state = env.reset(index, exceed_vm=1)
    # done = False
    # while not done:
    #     action = first_fit(state)
    #     state, _, done = env.step(action)
    # N_vm = env.get_attr('length_vm')
    # N_vm += 40
    # print(N_vm)

    # 2. Testing
    # env = copy.deepcopy(env)
    state = env.reset(index, N_vm=N_vm)
    done = False
    while not done:
        action = agent(state)
        state, _, done = env.step(action)

    return env.get_attr('total_wait_time')

def main():
    # Environment parameters
    N = 5  # Number of servers
    cpu = 40  # Total CPU per NUMA
    mem = 90  # Total memory per NUMA

    env = SchedEnv(N, cpu, mem, DATA_PATH, 10)
    first_fit = get_fit_func(0, cpu, mem)  # First Fit agent
    bal_fit = get_fit_func(2, cpu, mem)    # Balance Fit agent

    # Run multiple episodes
    res_f = []  # Results for First Fit
    res_b = []  # Results for Balance Fit
    # N_vms = []
    args1 = [(env, first_fit, valid_inds[episode], valid_Nvms[episode]) for episode in range(num_episodes)]
    args2 = [(env, bal_fit, valid_inds[episode], valid_Nvms[episode]) for episode in range(num_episodes)]
    with Pool(processes=6) as pool:
        res_f = pool.starmap(run_episode, args1)
        res_b = pool.starmap(run_episode, args2)
        # results = pool.map(run_episode_pool, range(num_episodes))
    # for episode in trange(num_episodes):
    #     index = valid_inds[episode]  # Select a valid test index
    #     N_vm = valid_Nvms[episode]
    #     res_f.append(run_episode(env, first_fit, index, N_vm))
    #     # N_vms.append(result[1])
    #     res_b.append(run_episode(env, bal_fit, index, N_vm))
    print(f"first fit: {trimmed_mean(res_f)}")
    print(f"first fit: {np.mean(res_f)}")
    # N_vms = np.array(N_vms)
    # np.save("data/valid_random_Nvm.npy", N_vms)
    print(f"balance fit: {trimmed_mean(res_b)}")
    print(f"balance fit: {np.mean(res_b)}")
    '''
    first fit: 89822.5225
    first fit: 362634.624
    balance fit: 86009.90875
    balance fit: 336486.767
    '''

if __name__ == "__main__":
    main()
