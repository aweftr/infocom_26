# %%
# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import os
import random
import time
from dataclasses import dataclass


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils.rnn import pad_sequence

from schedgym.sched_env_minusage import SchedEnv, getData
from ppo_agent import *
from baseline_agent import get_fit_func
from common import linear_decay, trimmed_mean, EarlyStopping
valid_inds = np.load('data/valid_random_time_150.npy')
valid_num = 50

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
    valide_patience: int = 20
    """Early stop patience"""

    # Environment specific arguments
    MAX_PM_NUM: int = 100
    """Maximum number of PMs, avoid too long input tensor"""
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
    N_vm: int = 5000
    """The VM sequecne length"""

    # Algorithm specific arguments
    total_timesteps: int = 500000
    """total timesteps of the experiments"""
    learning_rate: float = 1e-3
    """the learning rate of the optimizer"""
    num_envs: int = 8
    """the number of parallel game environments"""
    num_steps: int = 256
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def val(valid_envs: MyVectorEnvWithIndex, agent: Agent):
    obs, avail = valid_envs.reset(valid_inds[:valid_num])
    
    total_pm_usage = []
    infos = None
    with torch.no_grad():
        for i in range(100):
            obs = np.array(obs, dtype=np.object_)
            avail = np.array(avail, dtype=np.object_)

            vm_features, pm_padded, pm_mask, action_mask = getFeaturesMasks(obs, avail, pm_feature_num)
            # if pm_mask.shape[1] == 201:
            #     breakpoint()
            # print(pm_mask.shape)
            actions = agent.get_action(vm_features, pm_padded, pm_mask, action_mask)
            for idx, act in enumerate(actions.cpu().numpy()):
                if (avail[idx].shape[0] - 1) // 2 >= args.MAX_PM_NUM:
                    actions[idx] = np.random.choice(np.where(avail[idx])[0][1:])
            obs, rewards, truncated, avail, infos = valid_envs.step(actions)

    for i in range(valid_num):
        total_pm_usage.append(infos[i]["total_pm_usage"])
    # res_ppo = []

    return trimmed_mean(total_pm_usage)
# %%
# args = tyro.cli(Args)
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

# TRY NOT TO MODIFY: seeding
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = args.torch_deterministic

device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

# env = SchedEnv(args.PM_number, args.PM_cpu_oneNUME, args.PM_mem_oneNUME, args.data_path, args.double_thr)

data = getData(args.data_path, args.double_thr)
def make_env():
    return SchedEnv(args.PM_cpu_oneNUME, args.PM_mem_oneNUME, data, args.double_thr, random_reset=True)

def make_valid_env():
    return SchedEnv(args.PM_cpu_oneNUME, args.PM_mem_oneNUME, data, args.double_thr)
# %%

envs = MyVectorEnvLt(make_env, args.num_envs, N_vm=args.N_vm, lt_thre=args.lt_thre)

valid_envs = MyVectorEnvWithIndexLt(make_valid_env, valid_num, N_vm=100, lt_thre=args.lt_thre)
# obs, avails = envs.reset()
# while True:
#     actions = envs.sample_action(avails)
#     obs, rewards, truncated, avails, infos = envs.step(actions)
#     print(actions, truncated, rewards)
#     done = len(np.where(truncated)[0]) > 0
#     if done:
#         break

vm_feature_num = 4
# VM: cpu, mem, split, class{0, 1}
pm_feature_num = 3
# PM: cpu, mem, class{0, 1}
agent = TransformerPPOAgent(vm_feature_num, pm_feature_num).to(device)
agent.load_state_dict(torch.load(os.path.join("runs", "ppoImp_BCpretrain__1__1754016294", "model.pth"))) # Clair

# agent.load_state_dict(torch.load(os.path.join("runs", "ppoImp_BCpretrain__1__1753989075", "model.pth"))) # FF
optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

es = EarlyStopping(patience=args.valide_patience, path=f"runs/{run_name}/model.pth", delta=-1)
# %%

# ALGO Logic: Storage setup
# obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
actions = torch.zeros((args.num_steps, args.num_envs)).to(device)
# avails = torch.zeros((args.num_steps, args.num_envs, envs.single_action_space.n), dtype=torch.bool).to(device)
logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
values = torch.zeros((args.num_steps, args.num_envs)).to(device)


# TRY NOT TO MODIFY: start the game
global_step = 0
start_time = time.time()
next_obs, next_avail = envs.reset(seed=args.seed)
for idx, env in enumerate(envs.envs):
    print(f"Init index of env {idx}: {env.init_index}")
# next_obs = torch.Tensor(next_obs).to(device)
# next_avail = torch.tensor(next_avail, dtype=torch.bool).to(device)
next_done = torch.zeros(args.num_envs).to(device)

def getFeaturesMasks(obs: np.array, avails, pm_feature_num):
    vm_features = torch.Tensor(np.array(obs[:, 0].tolist(), dtype=float)).to(device)
    # vm_features = torch.Tensor(b_obs[mb_inds][:, 0]).to(device)
    pm_tensor_list = [torch.Tensor(pm) for pm in obs[:, 1]]
    # breakpoint()
    pm_padded = pad_sequence(pm_tensor_list, batch_first=True).reshape(obs.shape[0], -1, pm_feature_num).to(device)
    pm_mask = torch.tensor([
        [1] * (pm.shape[0] // pm_feature_num + 1) + [0] * (pm_padded.shape[1] - pm.shape[0] // pm_feature_num)
        for pm in pm_tensor_list
    ], dtype=torch.bool).to(device)

    action_mask = np.zeros((obs.shape[0], pm_padded.shape[1] + 1))
    for idx, avail in enumerate(avails):
        action_mask[idx][:len(avail)] = avail
    action_mask = torch.tensor(action_mask, dtype=torch.bool).to(device)

    return vm_features, pm_padded, pm_mask, action_mask

# %%
for iteration in range(1, args.num_iterations + 1):
    # Consider the value of the truncated observation!
    done_values = {}
    obs = []
    avails = []
    # Annealing the rate if instructed to do so.
    if args.anneal_lr:
        frac = 1.0 - (iteration - 1.0) / args.num_iterations
        lrnow = frac * args.learning_rate
        optimizer.param_groups[0]["lr"] = lrnow

    for step in range(0, args.num_steps):
        global_step += args.num_envs
        obs.append(next_obs)
        avails.append(next_avail)
        dones[step] = next_done

        tobs = np.array(next_obs, dtype=np.object_)
        tavails = np.array(next_avail, dtype=np.object_)
        vm_features, pm_padded, pm_mask, action_mask = getFeaturesMasks(tobs, tavails, pm_feature_num)
        # ALGO LOGIC: action logic
        with torch.no_grad():
            action, logprob, _, value = agent.get_action_and_value(vm_features, pm_padded, pm_mask, action_mask)
            values[step] = value.flatten()
        for idx, act in enumerate(action.cpu().numpy()):
            if (tavails[idx].shape[0] - 1) // 2 >= args.MAX_PM_NUM:
                action[idx] = np.random.choice(np.where(next_avail[idx])[0][1:])
        actions[step] = action
        logprobs[step] = logprob

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, reward, truncations, next_avail, infos = envs.step(action.cpu().numpy())
        # next_done = np.logical_or(terminations, truncations)
        rewards[step] = torch.tensor(reward).to(device).view(-1)
        
        next_done = torch.zeros(args.num_envs).to(device)
        
        for idx, info in enumerate(infos):
            if "done_info" in info:
                print(f"global_step={global_step}, total_pm_usage={info['total_pm_usage']}")
                if step not in done_values.keys():
                    done_values[step] = torch.zeros(args.num_envs).to(device)
                tobs = np.array(flatten_observation_lt(info["done_info"], args.lt_thre, info["pm_type"]), dtype=np.object_).reshape(1, 2)
                tavail = info["done_info"]["avail"].reshape(1, -1)
                vm_features, pm_padded, pm_mask, action_mask = getFeaturesMasks(tobs, tavail, pm_feature_num)
                with torch.no_grad():
                    done_values[step][idx] = agent.get_value(vm_features, pm_padded, pm_mask)
                next_done[idx] = 1
                writer.add_scalar("charts/total_pm_usage", info['total_pm_usage'], global_step)
    
    # bootstrap value if not done
    with torch.no_grad():
        tobs = np.array(next_obs, dtype=np.object_)
        tavails = np.array(next_avail, dtype=np.object_)
        vm_features, pm_padded, pm_mask, action_mask = getFeaturesMasks(tobs, tavails, pm_feature_num)
        next_value = agent.get_value(vm_features, pm_padded, pm_mask).reshape(1, -1)
        
        advantages = torch.zeros_like(rewards).to(device)
        lastgaelam = 0
        for t in reversed(range(args.num_steps)):
            donevalues = torch.zeros(args.num_envs).to(device)
            if t == args.num_steps - 1:
                nextnonterminal = 1.0 - next_done
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - dones[t + 1]
                nextvalues = values[t + 1]
            if torch.any((1 - nextnonterminal)):
                donevalues = done_values[t]
            
            delta = rewards[t] + args.gamma * nextvalues * nextnonterminal + args.gamma * donevalues * (1 - nextnonterminal) - values[t]
            advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * lastgaelam
        returns = advantages + values

    # flatten the batch
    
    b_obs = np.array(obs, dtype=np.object_).reshape(-1, 2)
    b_logprobs = logprobs.reshape(-1)
    b_actions = actions.reshape(-1)
    b_avails = np.array(avails, dtype=np.object_).reshape(-1)
    b_advantages = advantages.reshape(-1)
    b_returns = returns.reshape(-1)
    b_values = values.reshape(-1)

    # Optimizing the policy and value network
    b_inds = np.arange(args.batch_size)
    clipfracs = []
    for epoch in range(args.update_epochs):
        np.random.shuffle(b_inds)
        for start in range(0, args.batch_size, args.minibatch_size):
            end = start + args.minibatch_size
            mb_inds = b_inds[start:end]

            vm_features, pm_padded, pm_mask, action_mask = getFeaturesMasks(b_obs[mb_inds], b_avails[mb_inds], pm_feature_num)
            _, newlogprob, entropy, newvalue = agent.get_action_and_value(vm_features, pm_padded, pm_mask, action_mask=action_mask, action=b_actions.long()[mb_inds])
            logratio = newlogprob - b_logprobs[mb_inds]
            ratio = logratio.exp()

            with torch.no_grad():
                # calculate approx_kl http://joschu.net/blog/kl-approx.html
                old_approx_kl = (-logratio).mean()
                approx_kl = ((ratio - 1) - logratio).mean()
                clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

            mb_advantages = b_advantages[mb_inds]
            if args.norm_adv:
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

            # Policy loss
            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            # Value loss
            newvalue = newvalue.view(-1)
            if args.clip_vloss:
                v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                v_clipped = b_values[mb_inds] + torch.clamp(
                    newvalue - b_values[mb_inds],
                    -args.clip_coef,
                    args.clip_coef,
                )
                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()
            else:
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

            entropy_loss = entropy.mean()
            loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
            optimizer.step()

        if args.target_kl is not None and approx_kl > args.target_kl:
            break

    y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
    var_y = np.var(y_true)
    explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

    # TRY NOT TO MODIFY: record rewards for plotting purposes
    writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
    writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
    writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
    writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
    writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
    writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
    writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
    writer.add_scalar("losses/explained_variance", explained_var, global_step)
    print("SPS:", int(global_step / (time.time() - start_time)))
    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
    if (iteration + 1) % args.validate_interval == 0:
        val_loss = val(valid_envs, agent)
        # breakpoint()
        writer.add_scalar("val Loss", val_loss, global_step)
        es(val_loss, agent)
        if es.early_stop:
            print("Early stop! Val loss: {}".format(es.val_loss_min))
            break
        else:
            print("\tVal loss: {}".format(val_loss))

envs.close()
writer.close()
# torch.save(agent.state_dict(), f"runs/{run_name}/model.pth")