# %%
import pandas as pd
import numpy as np
import gymnasium as gym
from queue import PriorityQueue
from copy import deepcopy


def getData(path, double_thr=1e10):
    csv_data = pd.read_csv(path)
    data = csv_data.to_dict("records")
    for item in data:
        item["at"] = int(item["at"])
        item["lt"] = int(item["lt"])
        item["is_double"] = int(item["mem"] > double_thr)
    return data


class PM:
    def __init__(self, idx, cpu, mem):
        self.id = idx
        self.cpu = cpu
        self.mem = mem
        self.resources = np.full((2, 2), [self.cpu, self.mem], dtype=float)
        self.stored_vms = {}
        self.recent_usage_time = 0
        self.pm_type = None

    def reset(self):
        self.resources = np.full((2, 2), [self.cpu, self.mem], dtype=float)
        self.stored_vms = {}
        self.recent_usage_time = 0

    def handle(self, action, request):
        # action > 0
        numa = (action - 1) % 2
        if request["is_double"]:
            new_resources = self.resources - np.array(
                [request["cpu"] / 2, request["mem"] / 2]
            )
            if np.any(new_resources < 0):
                raise ValueError("Insufficient resources for allocation")
            self.resources = new_resources
        else:
            new_resources = self.resources[numa] - np.array(
                [request["cpu"], request["mem"]]
            )
            if np.any(new_resources < 0):
                raise ValueError("Insufficient resources for allocation")
            self.resources[numa] = new_resources
        # Store VM information
        self.stored_vms[request["vmid"]] = (self.id, numa, request["is_double"], request["at"], request["lt"])
        self.recent_usage_time = request["at"]

    def is_empty(self):
        return np.all(
            self.resources == np.array([[self.cpu, self.mem], [self.cpu, self.mem]])
        )

    def updatetype(self, split, ctime):
        if self.pm_type is None:
            return None
        
        vmtypes = []
        # 1 is long, 0 is short
        for vm in self.stored_vms:
            last_time = self.stored_vms[vm][4] - (ctime - self.stored_vms[vm][3])
            vmtypes.append(1 if last_time > split else 0)
        vmtypes = np.array(vmtypes)
        if np.any(vmtypes):
            self.pm_type = 1
            return 1
        else:
            self.pm_type = 0
            return 0


    def delete(self, request):
        if request["vmid"] not in self.stored_vms:
            raise ValueError("Delete unexist VM")

        serverid, numa, is_double, *rest = self.stored_vms.pop(request["vmid"])
        if is_double:
            self.resources += np.array([request["cpu"] / 2, request["mem"] / 2])
        else:
            self.resources[numa] += np.array([request["cpu"], request["mem"]])

    def check_request(self, request):
        req = np.array([request["cpu"], request["mem"]]).astype(float)
        if request["is_double"]:
            req /= 2
            return np.all(np.all(self.resources >= req, axis=1), axis=0).repeat(2)
        else:
            return np.all(self.resources >= req, axis=1).reshape(-1)

    def __repr__(self):
        return f"PM id:{self.id}, resources:{self.resources.reshape(-1)}, VMs:{self.stored_vms}"


# # test
# a = PM(0, 40, 90)
# vm = {"vmid": 0, "cpu": 30, "mem": 60, "is_double": 1}
# a.handle(1, vm)

# a = PM(0, 40, 90)
# vm = {"vmid": 0, "cpu": 30, "mem": 60, "is_double": 0}
# a.handle(1, vm)
# vm2 = {"vmid": 1, "cpu": 30, "mem": 60, "is_double": 0}
# print(a.check_request(vm2))

# a = PM(0, 40, 90)
# vm = {"vmid": 0, "cpu": 30, "mem": 60, "is_double": 1}
# a.handle(1, vm)
# vm2 = {"vmid": 1, "cpu": 30, "mem": 60, "is_double": 0}
# print(a.check_request(vm2))


# a = PM(0, 40, 90)
# vm = {"vmid": 0, "cpu": 30, "mem": 60, "is_double": 1}
# a.handle(1, vm)
# vm2 = {"vmid": 1, "cpu": 30, "mem": 60, "is_double": 1}
# print(a.check_request(vm2))
# %%
class Cluster:
    def __init__(self, cpu, mem):
        self.cpu = cpu
        self.mem = mem
        self.active_pms = []
        self.stored_vms = {}
        self.current_pm_index = 0
        self.reset()

    def reset(self):
        self.active_pms = []
        self.stored_vms = {}
        self.current_pm_index = 0
        pm = PM(self.current_pm_index, self.cpu, self.mem)
        self.active_pms.append(pm)
        self.current_pm_index += 1

    def handle(self, action, request):
        req = deepcopy(request)
        if action > len(self.active_pms) * 2:
            raise Exception("Invalid action to unknown PM!")
        if action == 0:
            # create a new pm to fit the request
            self.create_pm()
            # always put the request to the first numa
            # (There is no difference between first and second numa for an empty PM)
            self.active_pms[-1].handle(1, req)
            req["pmid"] = self.active_pms[-1].id
        else:
            # action in [1, ...], minus 1 to get the corresponding index
            self.active_pms[(action - 1) // 2].handle(action, req)
            req["pmid"] = self.active_pms[(action - 1) // 2].id
        self.stored_vms[req["vmid"]] = req

    def delete(self, vmid):
        req = self.stored_vms[vmid]
        deleted = False
        for pm in self.active_pms:
            if pm.id == req["pmid"]:
                pm.delete(req)
                if pm.is_empty():
                    self.active_pms.remove(pm)
                deleted = True
        if not deleted:
            raise Exception("VM not found in current active PMs!")
        self.stored_vms.pop(vmid)

    def create_pm(self):
        pm = PM(self.current_pm_index, self.cpu, self.mem)
        self.active_pms.append(pm)
        self.current_pm_index += 1

    def describe(self):
        pm_states = []
        for i in self.active_pms:
            pm_states.append(i.resources)
        return np.array(pm_states).reshape(-1, 2)

    def check_request(self, request):
        avail = []
        for pm in self.active_pms:
            avail.append(pm.check_request(request))
        avail = np.array(avail).reshape(-1)
        # Index 0 means open a new PM to fit the request
        # so this action is always true
        avail = np.insert(avail, 0, True)
        return avail

    def __repr__(self):
        return f"{self.active_pms}"


# # test
# a1 = PM(1, 40, 90)
# c = Cluster(40, 90)
# print(c.describe())
# c.active_pms.append(a1)
# print(c.describe())
# vm = {"vmid": 0, "cpu": 40, "mem": 60, "is_double": 0}
# print(c.check_request(vm))
# vm = {"vmid": 0, "cpu": 50, "mem": 60, "is_double": 0}
# print(c.check_request(vm))
# vm = {"vmid": 0, "cpu": 80, "mem": 60, "is_double": 1}
# print(c.check_request(vm))
# %%
class SchedEnv(gym.Env):
    def __init__(
        self,
        cpu,
        mem,
        data,
        double_thr=10,
        reward_type="basic",
        reward_weight=0,
        random_reset=False,
    ):
        super(SchedEnv, self).__init__()
        self.cpu = cpu
        self.mem = mem
        self.cluster = Cluster(cpu, mem)
        self.data = data
        self.reward_type = reward_type
        self.reward_weight = reward_weight
        self.total_pm_usage = 0
        self.maximum_pm_num = 0
        self.random_reset = random_reset
        if self.random_reset:
            print("Warning: Random reset is on, index is of no usage.")

    def reset(self, index=0, N_vm=0, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        if self.random_reset:
            if index != 0:
                print("Warning! Index and random_reset should not use simutaneously.")
            index = self.np_random.integers(0, 100000, dtype=np.int32)
        if N_vm == 0:
            # N_vm is the number of vm to be allocated in this reset
            raise Exception("You should set N_vm to be positive")
        self.cluster.reset()
        self.init_index = index
        self.index = index
        self.N_vm = N_vm

        self.t = self.data[self.index]["at"]  # Current time
        self.cnt = 0  # Number of created VMs
        self.vm_end_times = PriorityQueue()
        self.bal_score = 0  # Balance score for the cluster
        self.total_pm_usage = 0
        self.maximum_pm_num = 0
        return self.get_input()

    def step(self, action):
        request = self.data[self.index]
        self.t = request["at"]
        # print(f"current request: {request}")
        # print(f"current time: {self.t}")
        if self.cluster.check_request(request)[action] == 0:
            raise ValueError("Agent selected an unavailable NUMA")

        self.cluster.handle(action, request)
        # print(self.cluster)
        if len(self.cluster.active_pms) > self.maximum_pm_num:
            self.maximum_pm_num = len(self.cluster.active_pms)
        end_time = self.t + request["lt"]
        self.vm_end_times.put((end_time, self.index))
        self.cnt += 1
        self.index += 1

        done = False
        if self.index >= len(self.data) - 1:
            done = True
        else:
            done = self.cnt >= self.N_vm

        # scale the reward since it is too large in some point.
        reward = -(self._advance_time() / 1000)
        next_state = self.get_input()

        return next_state, reward, done

    def _advance_time(self):
        next_vm = self.data[self.index]
        next_vm_time = next_vm["at"]
        pm_usage_thistime = 0

        # print("advance time")
        # Handle all VMs that finish before the next VM arrives
        while (not self.vm_end_times.empty()) and (
            self.vm_end_times.queue[0][0] <= next_vm_time
        ):
            end_time, vmid = self.vm_end_times.get()
            current_time = self.t
            self.t = end_time
            # ended_vm = self.data[vmid]
            # print(f"current time: {current_time}, end_time: {end_time}")
            pm_usage_thistime += (end_time - current_time) * len(
                self.cluster.active_pms
            )
            self.cluster.delete(vmid)
        if self.t != next_vm_time:
            pm_usage_thistime += (next_vm_time - self.t) * len(self.cluster.active_pms)
        self.total_pm_usage += pm_usage_thistime
        # print(f"Advance complete, pm_usage: {pm_usage_thistime}\n")
        return pm_usage_thistime

    def get_input(self):
        request = self.data[self.index]
        return {
            "obs": self.cluster.describe(),
            "feat": np.array([request["cpu"], request["mem"], request["is_double"]]),
            "avail": self.cluster.check_request(request),
            "lt": request["lt"]
        }


# data = getData("../data/Huawei-East-1-lt.csv", 10)
# a = SchedEnv(40, 90, data, 10, random_reset=True)

# N_vms = [100, 200, 500, 1000, 2000, 3000, 5000, 7500, 10000]
# # N_vms = [100, 200, 500, 1000]
# VM2PM = {}
# for N_vm in N_vms:
#     max_pms = []
#     for i in range(50):
#         state = a.reset(N_vm=N_vm)
#         # print(a.index)
#         done = False
#         while not done:
#             avail = state["avail"][1:]
#             if np.any(avail):
#                 action = np.random.choice(np.where(avail)[0] + 1)
#             else:
#                 action = 0
#             state, reward, done = a.step(action)
#         max_pms.append(a.maximum_pm_num)
#     print(np.array(max_pms).mean())
#     VM2PM[N_vm] = np.array(max_pms).mean()
# print(VM2PM)
# # {100: np.float64(2.42), 200: np.float64(2.66), 500: np.float64(5.58), 1000: np.float64(9.14), 2000: np.float64(13.38), 3000: np.float64(17.72), 5000: np.float64(23.68), 7500: np.float64(28.04), 10000: np.float64(32.38)}
# # %%
# import matplotlib.pyplot as plt
# plt.plot(list(VM2PM.values()))
# plt.xticks(np.arange(len(N_vms)), N_vms)
# plt.xlabel("VMs")
# plt.ylabel("PMs")
# plt.savefig("VM2PM.png")
# # plt.show()
# # %%
