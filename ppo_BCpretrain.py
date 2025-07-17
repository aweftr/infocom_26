''' Evaluate the results of First Fit and Balance Fit '''

import numpy as np
from schedgym.sched_env import SchedEnv 
from baseline_agent import get_fit_func
from tqdm import trange
from common import trimmed_mean, EarlyStopping
from ppo_agent import Agent, flatten_observation, MyVectorEnv
import os
import random
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import time
from dataclasses import dataclass
DATA_PATH = "data/Huawei-East-1-lt.csv"

use_gpu = False
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
    validate_interval: int = 20
    """Validate the agent per interval to avoid overfitting by early stopping"""
    valide_patience: int = 30
    """Early stop patience"""

    # Environment specific arguments
    PM_number: int = 5
    """The total number of PM in the cluster"""
    PM_cpu_oneNUME: int = 40
    """The cpu capacity of PM in one NUMA"""
    PM_mem_oneNUME: int = 90
    """The mem capacity of PM in one NUMA"""
    double_thr: int = 10
    """If request mem >= double_thr, it should be scheduled to two NUMAs"""
    data_path: str = "data/Huawei-East-1-lt.csv"
    """The input data path"""
    exceed_vm: int = 40
    """Allocate another exceed_vm after the first VM has to wait"""

    total_timesteps: int = 500000
    """total timesteps of the experiments"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    learning_rate: float = 1e-4
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

def run_episode(env: SchedEnv, agent, index, N_vm):
    # 2. Testing
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

def make_env():
    return SchedEnv(args.PM_number, args.PM_cpu_oneNUME, args.PM_mem_oneNUME, args.data_path, args.double_thr, random_reset=True)

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

envs = MyVectorEnv(make_env, args.num_envs, args.exceed_vm)
# env = SchedEnv(args.PM_number, args.PM_cpu_oneNUME, args.PM_mem_oneNUME, args.data_path, args.double_thr, random_reset=True)
# env.reset(N_vm=10)
# breakpoint()
agent = Agent(envs.envs[0]).to(device)
bal_fit = get_fit_func(2, args.PM_cpu_oneNUME, args.PM_mem_oneNUME)    # Balance Fit agent
optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

es = EarlyStopping(patience=args.valide_patience, path=f"runs/{run_name}/model.pth", delta=-0.1)

# collect data
obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
avails = torch.zeros((args.num_steps, args.num_envs, envs.single_action_space.n), dtype=torch.bool).to(device)
dones = torch.zeros((args.num_steps, args.num_envs)).to(device)

# TRY NOT TO MODIFY: start the game
global_step = 0
start_time = time.time()
next_obs, next_avail = envs.reset(seed=args.seed)
for idx, env in enumerate(envs.envs):
    print(f"Init index of env {idx}: {env.init_index}")
next_obs = torch.Tensor(next_obs).to(device)
next_avail = torch.tensor(next_avail, dtype=torch.bool).to(device)
next_done = torch.zeros(args.num_envs).to(device)

def ppotorchToNumpy(obs, avails):
    obs = obs.cpu().numpy()
    avails = avails.cpu().numpy()
    actions = []
    for i in range(args.num_envs):
        tobs = obs[i][:-3].reshape(-1, 2, 2)
        tfeat = obs[i][-3:]
        tavail = avails[i]
        o = {"obs": tobs, "feat": tfeat, "avail":tavail}
        # breakpoint()
        action = bal_fit(o)
        actions.append(action)
    return np.array(actions)

for iteration in range(args.num_iterations):
    if args.anneal_lr:
        frac = 1.0 - (iteration - 1.0) / args.num_iterations
        lrnow = frac * args.learning_rate
        optimizer.param_groups[0]["lr"] = lrnow

    for step in range(0, args.num_steps):
        global_step += args.num_envs
        obs[step] = next_obs
        avails[step] = next_avail
        dones[step] = next_done

        bal_actions = ppotorchToNumpy(next_obs, next_avail)
        actions[step] = torch.tensor(bal_actions).to(device)

        next_obs, reward, truncations, avail, infos = envs.step(bal_actions)
        next_obs = torch.Tensor(next_obs).to(device)
        next_avail = torch.tensor(avail).to(device)
        next_done = torch.zeros(args.num_envs).to(device)

        for idx, info in enumerate(infos):
            if "done_info" in info:
                print(f"global_step={global_step}, episide_wait_time={info['total_wait_time']}")
                writer.add_scalar("charts/episodic_wait_time", info['total_wait_time'], global_step)
    # flatten the batch
    b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
    b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
    b_avails = avails.reshape((-1, envs.single_action_space.n))
    # Optimizing the policy and value network
    b_inds = np.arange(args.batch_size)
    clipfracs = []
    for epoch in range(args.update_epochs):
        np.random.shuffle(b_inds)
        for start in range(0, args.batch_size, args.minibatch_size):
            end = start + args.minibatch_size
            mb_inds = b_inds[start:end]

            _, logprob, entropy, value = agent.get_action_and_value(b_obs[mb_inds], action=b_actions.long()[mb_inds], avail=b_avails[mb_inds])
            loss = -logprob.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # breakpoint()
    writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
    writer.add_scalar("losses/BCLoss", loss, global_step)
    if (iteration + 1) % args.validate_interval == 0:
        writer.add_scalar("val Loss", loss, global_step)
        es(loss, agent)
        if es.early_stop:
            print("Early stop! Val loss: {}".format(es.val_loss_min))
            break
        else:
            print("\tVal loss: {}".format(loss))

envs.close()
writer.close()
# torch.save(agent.state_dict(), f"runs/{run_name}/model.pth")