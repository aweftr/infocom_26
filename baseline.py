''' Evaluate the results of First Fit and Balance Fit '''

import numpy as np
# from schedgym.sched_env import SchedEnv 
from schedgym.sched_env_minusage import SchedEnv, getData
from baseline_agent import get_fit_func
from tqdm import trange
from multiprocessing import Pool
import copy
from common import trimmed_mean
DATA_PATH = "data/Huawei-East-1-lt.csv"
data = getData(DATA_PATH, 10)

# A. Validation Result
# valid_inds = np.load('data/valid_random_time_150.npy')
# valid_Nvms = np.load("data/valid_random_Nvm.npy")
# num_episodes = 150
# B. Test Result
valid_inds = np.load('data/test_random_time_1000.npy')
num_episodes = 1000
N_vm = 1000
num_processes = 12

first_fit = get_fit_func(0, 40, 90)
def run_episode(env: SchedEnv, agent, index, N_vm):
    # env = copy.deepcopy(env)
    state = env.reset(index, N_vm=N_vm)
    done = False
    while not done:
        action = agent(state)
        # print(state, action)
        # breakpoint()
        state, _, done = env.step(action)

    return env.total_pm_usage, env.maximum_pm_num

def run_episode_m2f(env: SchedEnv, agent, index, N_vm):
    # env = copy.deepcopy(env)
    state = env.reset(index, N_vm=N_vm)
    done = False
    while not done:
        action = agent(state, env.cluster)
        # print(state, action)
        # breakpoint()
        state, _, done = env.step(action)

    return env.total_pm_usage, env.maximum_pm_num

def run_episode_bal(env: SchedEnv, agent, index, N_vm):
    # env = copy.deepcopy(env)
    state = env.reset(index, N_vm=N_vm)
    done = False
    while not done:
        action = agent(state, "sum")
        # print(state, action)
        # breakpoint()
        state, _, done = env.step(action)

    return env.total_pm_usage, env.maximum_pm_num

def runEvaluate(env, agent, run_episode):
    args = [(env, agent, valid_inds[episode], N_vm) for episode in range(num_episodes)]
    with Pool(processes=num_processes) as pool:
        result = pool.starmap(run_episode, args)
        
    result = np.array(result)
    res = result[:, 0]
    maxi = result[:, 1]
    print(f"{agent.__name__} pm usage trimmed mean: {trimmed_mean(res)}")
    print(f"{agent.__name__} pm usage mean: {np.mean(res)}")
    print(f"{agent.__name__} max pm mean: {np.mean(maxi)}")


def main():
    # Environment parameters
    cpu = 40  # Total CPU per NUMA
    mem = 90  # Total memory per NUMA

    env = SchedEnv(cpu, mem, data, 10)
    first_fit = get_fit_func(0, cpu, mem)  # First Fit agent
    best_fit = get_fit_func(1, cpu, mem)
    # run_episode(env, best_fit, 0, N_vm)
    bal_fit = get_fit_func(2, cpu, mem)    # Balance Fit agent
    random_fit = get_fit_func(3, cpu, mem)
    m2f_fit = get_fit_func(4, cpu, mem)
    

    runEvaluate(env, first_fit, run_episode)
    # runEvaluate(env, best_fit, run_episode)
    # runEvaluate(env, random_fit, run_episode)
    # run_episode_m2f(env, m2f_fit, 0, N_vm)
    # run_episode_bal(env, bal_fit, 0, N_vm, "max")
    # runEvaluate(env, m2f_fit, run_episode_m2f)
    # runEvaluate(env, bal_fit, run_episode_bal)

    # Run multiple episodes
    # res_b = []  # Results for Balance Fit
    # N_vms = []
    # args1 = [(env, first_fit, valid_inds[episode], N_vm) for episode in range(num_episodes)]
    # args2 = [(env, best_fit, valid_inds[episode], N_vm) for episode in range(num_episodes)]
    # # args2 = [(env, bal_fit, valid_inds[episode], valid_Nvms[episode]) for episode in range(num_episodes)]
    # args3 = [(env, random_fit, valid_inds[episode], N_vm) for episode in range(num_episodes)]
    # with Pool(processes=6) as pool:
    #     resultsf = pool.starmap(run_episode, args1)
    #     # res_b = pool.starmap(run_episode, args2)
    #     resultsr = pool.starmap(run_episode, args3)
        # results = pool.map(run_episode_pool, range(num_episodes))
    # for episode in trange(num_episodes):
    #     index = valid_inds[episode]  # Select a valid test index
    #     N_vm = valid_Nvms[episode]
    #     res_f.append(run_episode(env, first_fit, index, N_vm))
    #     # N_vms.append(result[1])
    #     res_b.append(run_episode(env, bal_fit, index, N_vm))
    # resultsf = np.array(resultsf)
    # res = resultsf[:, 0]
    # maxi = resultsf[:, 1]
    # print(f"first fit pm usage trimmed mean: {trimmed_mean(res)}")
    # print(f"first fit pm usage mean: {np.mean(res)}")
    # print(f"first fit max pm mean: {np.mean(maxi)}")
    # # N_vms = np.array(N_vms)
    # # np.save("data/valid_random_Nvm.npy", N_vms)
    # # print(f"balance fit: {trimmed_mean(res_b)}")
    # # print(f"balance fit: {np.mean(res_b)}")
    # resultsr = np.array(resultsr)
    # res = resultsr[:, 0]
    # maxi = resultsr[:, 1]
    # print(f"random fit pm usage trimmed mean: {trimmed_mean(res)}")
    # print(f"random fit pm usage mean: {np.mean(res)}")
    # print(f"random fit max pm mean: {np.mean(maxi)}")
    '''
    first fit pm usage trimmed mean: 82931.2575
    first fit pm usage mean: 94520.816
    first fit max pm mean: 8.005
    balance_fit_max pm usage trimmed mean: 85119.0975
    balance_fit_max pm usage mean: 96639.953
    balance_fit_max max pm mean: 7.86
    balance_fit_sum pm usage trimmed mean: 85374.07875
    balance_fit_sum pm usage mean: 97015.014
    balance_fit_sum max pm mean: 7.903
    best_fit pm usage trimmed mean: 86592.45625
    best_fit pm usage mean: 99468.216
    best_fit max pm mean: 8.14
    random fit pm usage trimmed mean: 90808.88625
    random fit pm usage mean: 104433.836
    random fit max pm mean: 8.09
    movetofront_fit pm usage trimmed mean: 93487.46
    movetofront_fit pm usage mean: 107088.683
    movetofront_fit max pm mean: 8.272
    ppoBCpre pm usage trimmed mean: 83062.615
    ppoBCpre pm usage mean: 94714.161
    ppoBCpre max pm mean: 8.023
    '''

if __name__ == "__main__":
    main()
