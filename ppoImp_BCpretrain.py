''' PPO BC pretrain '''

import numpy as np
from schedgym.sched_env_minusage import SchedEnv, getData
from baseline_agent import get_fit_func, clairvoyant_ltfit_updatePM, first_fit
from tqdm import trange
from common import trimmed_mean, EarlyStopping
from ppo_agent import *
import os
import random
import torch
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.tensorboard import SummaryWriter
import time
from dataclasses import dataclass
import itertools
DATA_PATH = "data/Huawei-East-1-lt.csv"
data = getData(DATA_PATH, 10)

use_gpu = True
device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    validate_interval: int = 5
    """Validate the agent per interval to avoid overfitting by early stopping"""
    valide_patience: int = 10
    """Early stop patience"""

    # Environment specific arguments
    PM_cpu_oneNUME: int = 40
    """The cpu capacity of PM in one NUMA"""
    PM_mem_oneNUME: int = 90
    """The mem capacity of PM in one NUMA"""
    double_thr: int = 10
    """If request mem >= double_thr, it should be scheduled to two NUMAs"""
    lt_thre: int = 8000
    """Lifetime threshold to determine long or short class"""
    data_path: str = "data/Huawei-East-1-lt.csv"
    """The input data path"""
    N_vm: int = 1000
    """The VM sequecne length"""

    total_timesteps: int = 100000
    """total timesteps of the experiments"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    learning_rate: float = 1e-3
    """the learning rate of the optimizer"""
    num_envs: int = 4
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def make_env():
    return SchedEnv(args.PM_cpu_oneNUME, args.PM_mem_oneNUME, data, args.double_thr, random_reset=True)

# Environment parameters
args = Args()
args.batch_size = int(args.num_envs * args.num_steps)
args.minibatch_size = int(args.batch_size // args.num_minibatches)
args.num_iterations = args.total_timesteps // args.batch_size
run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"

writer = SummaryWriter(f"runs/{run_name}")
writer.add_text(
    "hyperparameters",
    "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = args.torch_deterministic

envs = MyVectorEnvLt(make_env, args.num_envs, args.N_vm, args.lt_thre)
# env = SchedEnv(args.PM_number, args.PM_cpu_oneNUME, args.PM_mem_oneNUME, args.data_path, args.double_thr, random_reset=True)
# env.reset(N_vm=10)

vm_feature_num = 4
# VM: cpu, mem, split, class{0, 1}
pm_feature_num = 3
# PM: cpu, mem, class{0, 1}
agent = TransformerPPOAgent(vm_feature_num, pm_feature_num).to(device)
# first_fit = get_fit_func(0, args.PM_cpu_oneNUME, args.PM_mem_oneNUME)    # Frist Fit agent
optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

es = EarlyStopping(patience=args.valide_patience, path=f"runs/{run_name}/model.pth", delta=-0.01)

# collect data
actions = torch.zeros((args.num_steps, args.num_envs)).to(device)
dones = torch.zeros((args.num_steps, args.num_envs)).to(device)


# TRY NOT TO MODIFY: start the game
global_step = 0
start_time = time.time()
next_obs, next_avail = envs.reset(seed=args.seed)

for idx, env in enumerate(envs.envs):
    print(f"Init index of env {idx}: {env.init_index}")
next_done = torch.zeros(args.num_envs).to(device)

def getBCActions(envs):
    actions = []
    for i, env in enumerate(envs.envs):
        state = env.get_input()
        action = clairvoyant_ltfit_updatePM.clairvoyant_ltfit_binary(state, env, args.lt_thre)
        actions.append(action)
    return np.array(actions)

def getFFActions(envs):
    actions = []
    for i, env in enumerate(envs.envs):
        state = env.get_input()
        action = first_fit(state)
        actions.append(action)
    return np.array(actions)

def getFeaturesMasks(obs: np.array, avails, pm_feature_num):
    vm_features = torch.Tensor(np.array(obs[:, 0].tolist(), dtype=float)).to(device)
    # vm_features = torch.Tensor(b_obs[mb_inds][:, 0]).to(device)
    pm_tensor_list = [torch.Tensor(pm) for pm in obs[:, 1]]
    pm_padded = pad_sequence(pm_tensor_list, batch_first=True).reshape(args.minibatch_size, -1, pm_feature_num).to(device)
    pm_mask = torch.tensor([
        [1] * (pm.shape[0] // pm_feature_num + 1) + [0] * (pm_padded.shape[1] - pm.shape[0] // pm_feature_num)
        for pm in pm_tensor_list
    ], dtype=torch.bool).to(device)

    action_mask = np.zeros((args.minibatch_size, pm_padded.shape[1] + 1))
    for idx, avail in enumerate(avails):
        action_mask[idx][:len(avail)] = avail
    action_mask = torch.tensor(action_mask, dtype=torch.bool).to(device)

    return vm_features, pm_padded, pm_mask, action_mask


for iteration in range(args.num_iterations):
    obs = []
    avails = []
    if args.anneal_lr:
        frac = 1.0 - (iteration - 1.0) / args.num_iterations
        lrnow = frac * args.learning_rate
        optimizer.param_groups[0]["lr"] = lrnow

    for step in range(0, args.num_steps):
        global_step += args.num_envs
        obs.append(next_obs)
        avails.append(next_avail)
        # obs[step] = next_obs
        # avails[step] = next_avail
        dones[step] = next_done

        ff_actions = getBCActions(envs)
        
        actions[step] = torch.tensor(ff_actions).to(device)

        next_obs, reward, truncations, next_avail, infos = envs.step(ff_actions)
        next_done = torch.tensor(truncations).to(device)
        

        for idx, info in enumerate(infos):
            if "done_info" in info:
                print(f"global_step={global_step}, total_pm_usage={info['total_pm_usage']}")
                writer.add_scalar("charts/episodic_pm_usage", info['total_pm_usage'], global_step)
    # flatten the batch
    
    b_obs = np.array(obs, dtype=np.object_).reshape(-1, 2)
    b_actions = actions.reshape(-1)
    b_avails = np.array(avails, dtype=np.object_).reshape(-1)
    # breakpoint()

    # Optimizing the policy and value network
    b_inds = np.arange(args.batch_size)
    clipfracs = []
    for epoch in range(args.update_epochs):
        np.random.shuffle(b_inds)
        for start in range(0, args.batch_size, args.minibatch_size):
            end = start + args.minibatch_size
            mb_inds = b_inds[start:end]
            

            vm_features, pm_padded, pm_mask, action_mask = getFeaturesMasks(b_obs[mb_inds], b_avails[mb_inds], pm_feature_num)
            # print(b_actions.long()[mb_inds].max())
            # print(pm_padded.shape)
            # breakpoint()
            _, logprob, entropy, value = agent.get_action_and_value(vm_features, pm_padded, pm_mask, action_mask=action_mask, action=b_actions.long()[mb_inds])
            loss = -logprob.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # breakpoint()
    writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
    writer.add_scalar("losses/BCLoss", loss, global_step)
    if (iteration + 1) % args.validate_interval == 0:
        # writer.add_scalar("val Loss", loss, global_step)
        es(loss, agent)
        if es.early_stop:
            print("Early stop! Val loss: {}".format(es.val_loss_min))
            break
        else:
            print("\tVal loss: {}".format(loss))

envs.close()
# writer.close()
# torch.save(agent.state_dict(), f"runs/{run_name}/model.pth")