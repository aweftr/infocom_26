''' Evaluate the results of First Fit and Balance Fit '''

import numpy as np
from schedgym.sched_env_minusage import SchedEnv, getData
from baseline_agent import get_fit_func
from tqdm import trange
from common import trimmed_mean
from ppo_agent import *
import os
import torch
from torch.nn.utils.rnn import pad_sequence
import math
from copy import deepcopy
DATA_PATH = "data/Huawei-East-1-lt.csv"
data = getData(DATA_PATH, 10)

# A. Validation Result
# valid_inds = np.load('data/valid_random_time_150.npy')
# num_episodes = 150
# B. Test Result
valid_inds = np.load('data/test_random_time_1000.npy')
num_episodes = 1000
batch_size = 1000
N_vm = 1000
use_gpu = True
device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")


def make_env():
    return SchedEnv(cpu, mem, data, 10)

def getFeaturesMasks(obs: np.array, avails):
    vm_features = torch.Tensor(np.array(obs[:, 0].tolist(), dtype=float)).to(device)
    # vm_features = torch.Tensor(b_obs[mb_inds][:, 0]).to(device)
    pm_tensor_list = [torch.Tensor(pm) for pm in obs[:, 1]]
    # breakpoint()
    pm_padded = pad_sequence(pm_tensor_list, batch_first=True).reshape(obs.shape[0], -1, 2).to(device)
    pm_mask = torch.tensor([
        [1] * (pm.shape[0] // 2 + 1) + [0] * (pm_padded.shape[1] - pm.shape[0] // 2)
        for pm in pm_tensor_list
    ], dtype=torch.bool).to(device)

    action_mask = np.zeros((obs.shape[0], pm_padded.shape[1] + 1))
    for idx, avail in enumerate(avails):
        action_mask[idx][:len(avail)] = avail
    action_mask = torch.tensor(action_mask, dtype=torch.bool).to(device)

    return vm_features, pm_padded, pm_mask, action_mask



cpu = 40  # Total CPU per NUMA
mem = 90  # Total memory per NUMA


agent = TransformerPPOAgent(3, 2)
# path = os.listdir("runs")[-3]
# path = "ppo_train__1__1752493459" # vanilla train 9w+

# path = "ppo_BCpretrain__1__1753443805"
# PPO pm usage trimmed mean: 83069.1875
# PPO pm usage mean: 94741.432

path = "ppo_train__1__1753446038"
print(path)
agent.load_state_dict(torch.load(os.path.join("runs", path, "model.pth")))
agent.to(device)


num_batch = math.ceil(num_episodes / batch_size)

total_pm_usage = []
envs = MyVectorEnvWithIndex(make_env, num_episodes, N_vm)
obs, avail = envs.reset(valid_inds)

for batch in range(num_batch):
    s_index = batch_size * batch
    e_index = batch_size * (batch + 1)
    print(s_index, e_index)
    if e_index > num_episodes:
        e_index = num_episodes
    
    envs = MyVectorEnvWithIndex(make_env, batch_size, N_vm)

    obs, avail = envs.reset(valid_inds[s_index: e_index])
    with torch.no_grad():
        for i in trange(N_vm):
            obs = np.array(obs, dtype=np.object_)
            avail = np.array(avail, dtype=np.object_)

            vm_features, pm_padded, pm_mask, action_mask = getFeaturesMasks(obs, avail)
            actions = agent.get_action(vm_features, pm_padded, pm_mask, action_mask)
            obs, rewards, truncated, avail, infos = envs.step(actions)

    for i in range(e_index - s_index):
        total_pm_usage.append(infos[i]["total_pm_usage"])


# breakpoint()

# for idx, env in enumerate(envs.envs):
#     print(f"Init index of env {idx}: {env.init_index}")


print(f"PPO pm usage trimmed mean: {trimmed_mean(total_pm_usage)}")
print(f"PPO pm usage mean: {np.mean(total_pm_usage)}")

# agent.get_action(vm_features, pm_features, pm_mask, action_mask)

    # a = run_episode(env, agent, valid_inds[0], 1000)
    # breakpoint()
    # for i in os.listdir("runs")
    # Run multiple episodes
    # runEvaluate(env, run_episode)

    # res_ppo = []
    # # first_fit = get_fit_func(0, cpu, mem)  # First Fit agent
    # # args = [(env, agent, valid_inds[episode]) for episode in range(num_episodes)]
    # # with Pool(processes=8) as pool:
    # #     result = pool.starmap(run_episode, args)
    # for episode in trange(num_episodes):
    #     index = valid_inds[episode]  # Select a valid test index
    # #     # res_f.append(run_episode(env, first_fit, index))
    # #     # res_b.append(run_episode(env, bal_fit, index))
    #     res_ppo.append(run_episode(env, agent, index))
    # # print(f"first fit: {trimmed_mean(res_f)}")
    # # print(f"first fit: {np.mean(res_f)}")
    # # print(f"balance fit: {trimmed_mean(res_b)}")
    # # print(f"balance fit: {np.mean(res_b)}")
    # result = np.array(res_ppo)
    # res = result[:, 0]
    # maxi = result[:, 1]
    # print(f"ppo pm usage trimmed mean: {trimmed_mean(res)}")
    # print(f"ppo pm usage mean: {np.mean(res)}")
    # print(f"ppo max pm mean: {np.mean(maxi)}")

# if __name__ == "__main__":
#     main()
