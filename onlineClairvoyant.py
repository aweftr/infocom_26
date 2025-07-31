''' Evaluate the results of First Fit and Balance Fit '''

import numpy as np
import time
# from schedgym.sched_env import SchedEnv 
from schedgym.sched_env_minusage import SchedEnv, getData
from baseline_agent import clairvoyant_ltfit, get_fit_func, clairvoyant_ltfit_updatePM, clairvoyant_ltfit_smallBF,clairvoyant_ltfit_smallFF,clairvoyant_ltfit_improved
from tqdm import trange
from multiprocessing import Pool
import copy
from common import trimmed_mean
DATA_PATH = "data/Huawei-East-1-lt.csv"
# DATA_PATH = "data/Huawei-East-1-ltlogNormalNoise.csv"
# DATA_PATH = "data/Huawei-East-1-ltGaussianNoise.csv"
data = getData(DATA_PATH, 10)

# A. Validation Result
# valid_inds = np.load('data/valid_random_time_150.npy')
# valid_Nvms = np.load("data/valid_random_Nvm.npy")
# num_episodes = 150
# B. Test Result
valid_inds = np.load('data/test_random_time_1000.npy')
num_episodes = 1000
N_vm = 1000
lt_thre = 8000
num_processes = 14

def run_episode(env: SchedEnv, agent, index, N_vm):
    # env = copy.deepcopy(env)
    state = env.reset(index, N_vm=N_vm)
    done = False
    latency = []
    while not done:
        stime = time.time()
        action = agent(state)
        etime = time.time()
        latency.append(etime - stime)
        # print(state, action)
        # breakpoint()
        state, _, done = env.step(action)

    return env.total_pm_usage, env.maximum_pm_num, latency


def run_episode_online(env: SchedEnv, agent, index, N_vm, lt_thre):
    # env = copy.deepcopy(env)
    state = env.reset(index, N_vm=N_vm)
    done = False
    while not done:
        # action = agent(state)
        action = agent(state, env, lt_thre)
        # state contains obs, feat, avail
        # env is used to provide the corresponding PM type to the scheduler

        # breakpoint()
        # print(action, state["avail"], len(env.cluster.active_pms))
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


def runEvaluateOnline(env, agent, run_episode, lt_thre):
    args = [(env, agent, valid_inds[episode], N_vm, lt_thre) for episode in range(num_episodes)]
    with Pool(processes=num_processes) as pool:
        result = pool.starmap(run_episode, args)
        
    result = np.array(result)
    res = result[:, 0]
    maxi = result[:, 1]
    print(f"{agent.__name__} {lt_thre} pm usage trimmed mean: {trimmed_mean(res)}")
    print(f"{agent.__name__} {lt_thre} pm usage mean: {np.mean(res)}")
    print(f"{agent.__name__} {lt_thre} max pm mean: {np.mean(maxi)}")

def runEvaluate(env, agent, run_episode):
    args = [(env, agent, valid_inds[episode], N_vm) for episode in range(num_episodes)]
    with Pool(processes=8) as pool:
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
    # run_episode(env, first_fit, 0, N_vm)
    # print(env.total_pm_usage)
    bal_fit = get_fit_func(2, cpu, mem)    # Balance Fit agent
    random_fit = get_fit_func(3, cpu, mem)
    m2f_fit = get_fit_func(4, cpu, mem)
    # clairvoyant_fit = get_fit_func(5, cpu, mem)
    clairvoyant_fit = clairvoyant_ltfit_updatePM.clairvoyant_ltfit_binary
    # run_episode_online(env, clairvoyant_fit, 0, N_vm)
    # print(env.total_pm_usage)
    # run_episode_online(env, clairvoyant_fit, 0, N_vm, lt_thre)


    # _, _, latency = run_episode(env, first_fit, 0, N_vm)
    # print(latency)
    # runEvaluate(env, first_fit, run_episode)
    # runEvaluate(env, best_fit, run_episode)
    # runEvaluate(env, random_fit, run_episode)
    # runEvaluate(env, m2f_fit, run_episode_m2f)
    # runEvaluate(env, bal_fit, run_episode_bal)
    
    runEvaluateOnline(env, clairvoyant_fit, run_episode_online, lt_thre)
    # run_episode_m2f(env, m2f_fit, 0, N_vm)
    # run_episode_bal(env, bal_fit, 0, N_vm, "max")
    # print(len(env.cluster.active_pms))

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
    1000 VMs
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
    PPOBCpre_improv pm usage trimmed mean: 86735.93
    PPOBCpre_improv pm usage mean: 98217.527
    PPOBCpre_improv max pm mean: 8.134
    clairvoyant_ltfit (binary with no pm type update) 8000 pm usage trimmed mean: 137732.6375
    clairvoyant_ltfit 8000 pm usage mean: 151083.005
    clairvoyant_ltfit 8000 max pm mean: 12.218
    clairvoyant_ltfit_multi (multi with no pm type update) 8000 pm usage trimmed mean: 228245.51875
    clairvoyant_ltfit_multi 8000 pm usage mean: 241286.481
    clairvoyant_ltfit_multi 8000 max pm mean: 20.676
    clairvoyant_ltfit (binary with pm type update) 8000 pm usage trimmed mean: 94251.50875
    clairvoyant_ltfit 8000 pm usage mean: 105909.299
    clairvoyant_ltfit 8000 max pm mean: 8.897
    clairvoyant_ltfit_multi (multi with pm type update) 8000 pm usage trimmed mean: 181731.345
    clairvoyant_ltfit_multi 8000 pm usage mean: 193399.112
    clairvoyant_ltfit_multi 8000 max pm mean: 15.622
    clairvoyant_ltfit_binary (binary with small BF) 8000 pm usage trimmed mean: 143004.7425
    clairvoyant_ltfit_binary 8000 pm usage mean: 159449.173
    clairvoyant_ltfit_binary 8000 max pm mean: 13.275
    clairvoyant_ltfit_multi (multi with small BF) 8000 pm usage trimmed mean: 229991.9375
    clairvoyant_ltfit_multi 8000 pm usage mean: 242724.117
    clairvoyant_ltfit_multi 8000 max pm mean: 20.716
    clairvoyant_ltfit_binary (binary with small FF) 8000 pm usage trimmed mean: 137602.96
    clairvoyant_ltfit_binary 8000 pm usage mean: 151098.257
    clairvoyant_ltfit_binary 8000 max pm mean: 12.294
    clairvoyant_ltfit_multi (multi with small FF) 8000 pm usage trimmed mean: 229779.51125
    clairvoyant_ltfit_multi 8000 pm usage mean: 242544.092
    clairvoyant_ltfit_multi 8000 max pm mean: 20.691
    clairvoyant_ltfit_binary (binary with all improve) 8000 pm usage trimmed mean: 92649.09375
    clairvoyant_ltfit_binary 8000 pm usage mean: 104086.37
    clairvoyant_ltfit_binary 8000 max pm mean: 8.685
    clairvoyant_ltfit_multi (multi with all improve) 8000 pm usage trimmed mean: 179617.41625
    clairvoyant_ltfit_multi 8000 pm usage mean: 191582.431
    clairvoyant_ltfit_multi 8000 max pm mean: 15.492
    spane pm usage trimmed mean: 134885.62
    spane pm usage mean: 156715.25
    spane max pm mean: 9.69

    5000 VMs
    first_fit pm usage trimmed mean: 989993.88
    first_fit pm usage mean: 1126954.703
    first_fit max pm mean: 17.852
    best_fit pm usage trimmed mean: 1080088.4125
    best_fit pm usage mean: 1244430.842
    best_fit max pm mean: 17.899
    random_fit pm usage trimmed mean: 1195346.59
    random_fit pm usage mean: 1385110.395
    random_fit max pm mean: 18.943
    movetofront_fit pm usage trimmed mean: 1225053.76875
    movetofront_fit pm usage mean: 1426435.648
    movetofront_fit max pm mean: 19.466
    balance_fit_sum pm usage trimmed mean: 1062319.9525
    balance_fit_sum pm usage mean: 1196076.123
    balance_fit_sum max pm mean: 17.931
    balance_fit_max pm usage trimmed mean: 1064378.51625
    balance_fit_max pm usage mean: 1199403.008
    balance_fit_max max pm mean: 17.901
    clairvoyant_ltfit_binary (binary with no pm type update) 8000 pm usage trimmed mean: 1245889.575
    clairvoyant_ltfit_binary 8000 pm usage mean: 1377761.817
    clairvoyant_ltfit_binary 8000 max pm mean: 22.755
    clairvoyant_ltfit_multi (multi with no pm type update) 8000 pm usage trimmed mean: 1677237.8525
    clairvoyant_ltfit_multi 8000 pm usage mean: 1825191.369
    clairvoyant_ltfit_multi 8000 max pm mean: 33.694
    clairvoyant_ltfit_binary (binary with pm type update) 8000 pm usage trimmed mean: 974654.61625
    clairvoyant_ltfit_binary 8000 pm usage mean: 1120308.292
    clairvoyant_ltfit_binary 8000 max pm mean: 19.474
    clairvoyant_ltfit_multi (multi with pm type update) 8000 pm usage trimmed mean: 1350711.84875
    clairvoyant_ltfit_multi 8000 pm usage mean: 1507033.618
    clairvoyant_ltfit_multi 8000 max pm mean: 25.205
    clairvoyant_ltfit_binary (binary with small BF) 8000 pm usage trimmed mean: 1385569.4125
    clairvoyant_ltfit_binary 8000 pm usage mean: 1505418.061
    clairvoyant_ltfit_binary 8000 max pm mean: 26.627
    clairvoyant_ltfit_binary (binary with all improve) 8000 pm usage trimmed mean: 992326.97125
    clairvoyant_ltfit_binary 8000 pm usage mean: 1140917.409
    clairvoyant_ltfit_binary 8000 max pm mean: 18.66
    clairvoyant_ltfit_multi (multi with all improve) 8000 pm usage trimmed mean: 1336963.81375
    clairvoyant_ltfit_multi 8000 pm usage mean: 1494924.408
    clairvoyant_ltfit_multi 8000 max pm mean: 25.148

    PPOBCpre pm usage trimmed mean: 1042701.655
    PPOBCpre pm usage mean: 1195767.827
    PPOBCpre max pm mean: 18.509
    spane pm usage trimmed mean: 1961043.75
    spane pm usage mean: 2035200.6733333333
    spane max pm mean: 27.325

    10000 VMs
    first_fit pm usage trimmed mean: 2362719.755
    first_fit pm usage mean: 2554152.826
    first_fit max pm mean: 21.661
    best_fit pm usage trimmed mean: 2655321.7925
    best_fit pm usage mean: 2902472.336
    best_fit max pm mean: 21.497
    random_fit pm usage trimmed mean: 3011592.84125
    random_fit pm usage mean: 3362868.636
    random_fit max pm mean: 23.563
    movetofront_fit pm usage trimmed mean: 3138130.8475
    movetofront_fit pm usage mean: 3522246.574
    movetofront_fit max pm mean: 24.68
    balance_fit_sum pm usage trimmed mean: 2566810.23625
    balance_fit_sum pm usage mean: 2753541.405
    balance_fit_sum max pm mean: 21.791
    balance_fit_max pm usage trimmed mean: 2576599.3675
    balance_fit_max pm usage mean: 2767135.178
    balance_fit_max max pm mean: 21.905
    clairvoyant_ltfit (binary with pm update) 8000 pm usage trimmed mean: 2301357.02625
    clairvoyant_ltfit 8000 pm usage mean: 2525719.249
    clairvoyant_ltfit 8000 max pm mean: 23.006
    spane pm usage trimmed mean: 6241678.466666667
    spane pm usage mean: 6235237.033333333
    spane max pm mean: 43.858333333333334
    '''

if __name__ == "__main__":
    main()
