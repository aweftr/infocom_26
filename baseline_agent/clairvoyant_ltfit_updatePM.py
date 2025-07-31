# online clairvoyant scheduling (all lifetime is known) OA
# long and short
# cluster the VM based on their lifetime into several buckets and use firstfit to allocate each bucket
import numpy as np

def updateBinarytype(pm, split, ctime):
    if pm.pm_type is None and len(pm.stored_vms) == 0:
        return None
    
    vmtypes = []
    # 1 is long, 0 is short
    for vm in pm.stored_vms:
        # 5 is the stored ltpred, 3 is the arrival time
        last_time = pm.stored_vms[vm][5] - (ctime - pm.stored_vms[vm][3])
        vmtypes.append(1 if last_time > split else 0)
    vmtypes = np.array(vmtypes)
    if np.any(vmtypes):
        pm.pm_type = 1
        return 1
    else:
        pm.pm_type = 0
        return 0

def updateMultitype(pm, ctime):
    if pm.pm_type is None and len(pm.stored_vms) == 0:
        return None
    
    vmtypes = []
    # 0 - 11
    for vm in pm.stored_vms:
        # 5 is the stored ltpred, 3 is the arrival time
        last_time = pm.stored_vms[vm][5] - (ctime - pm.stored_vms[vm][3])
        vmtype = np.floor(np.log2(last_time / 15)).astype(int)
        if vmtype > 11: vmtype = 11
        vmtypes.append(vmtype)
    vmtypes = np.array(vmtypes)
    pm.pm_type = np.max(vmtypes)
    return np.max(vmtypes)
    
def clairvoyant_ltfit_binary(state, env, lt_thre):
    obs = state["obs"].copy()
    feat = state["feat"].copy()
    numa_avail = state["avail"][1:].copy()
    lt = state["lt"]
    ltpred = state["ltpred"]
    ctime = env.t
    # cluster the pm based on their lifetime (long or short currently!)
    # 1 for long and 0 for short
    request_is_split = feat[2]
    request_type = 1 if ltpred > lt_thre else 0

    len_buckest = 2
    pm_buckets = [[] for i in range(len_buckest)]

    for idx, pm in enumerate(env.cluster.active_pms):
        # split the pm by their type
        updateBinarytype(pm, lt_thre, env.t)
        # pm.updatetype(lt_thre, env.t)
        if np.any(numa_avail[[idx*2, idx*2+1]]):
            if pm.pm_type is None:
                # if there is one new available pm
                pm.pm_type = request_type
            pm_buckets[pm.pm_type].append([idx, pm])
            # if pm.pm_type == 1:
            #     pm_buckets[1].append([idx, pm])
            # else:
            #     pm_buckets[0].append([idx, pm])
    # breakpoint()
    if len(pm_buckets[request_type]) == 0:
        # no pm have the request type, create a new pm
        return 0
    else:
        # This is the first fit strategy
        first_numa = pm_buckets[request_type][0][0] * 2
        if request_is_split:
            # big vm allocate to avail pm directly
            # [0] action is to create new pm, need to plus 1
            return first_numa + 1
        else:
            # small vm one numa may be unavailable
            if numa_avail[first_numa]:
                return first_numa + 1
            else:
                return first_numa + 2

def clairvoyant_ltfit_multi(state, env, lt_thre):
    obs = state["obs"].copy()
    feat = state["feat"].copy()
    numa_avail = state["avail"][1:].copy()
    lt = state["lt"]
    ltpred = state["ltpred"]
    ctime = env.t
    # cluster the pm based on their lifetime (long or short currently!)
    # 1 for long and 0 for short
    request_is_split = feat[2]
    request_type = np.floor(np.log2(ltpred / 15)).astype(int)
    # 0-11 classification, /15 follow the setting of mlsys 2023 paper
    if request_type > 11: request_type = 11
    

    len_buckest = 12
    pm_buckets = [[] for i in range(len_buckest)]

    for idx, pm in enumerate(env.cluster.active_pms):
        # split the pm by their type
        # pm.updatetype(lt_thre, env.t)
        updateMultitype(pm, env.t)
        if np.any(numa_avail[[idx*2, idx*2+1]]):
            if pm.pm_type is None:
                # if there is one new available pm
                pm.pm_type = request_type
            pm_buckets[pm.pm_type].append([idx, pm])
            # if pm.pm_type == 1:
            #     pm_buckets[1].append([idx, pm])
            # else:
            #     pm_buckets[0].append([idx, pm])
    # breakpoint()
    if len(pm_buckets[request_type]) == 0:
        # no pm have the request type, create a new pm
        return 0
    else:
        # This is the first fit strategy
        first_numa = pm_buckets[request_type][0][0] * 2
        if request_is_split:
            # big vm allocate to avail pm directly
            # [0] action is to create new pm, need to plus 1
            return first_numa + 1
        else:
            # small vm one numa may be unavailable
            if numa_avail[first_numa]:
                return first_numa + 1
            else:
                return first_numa + 2
