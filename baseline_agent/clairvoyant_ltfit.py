# online clairvoyant scheduling (all lifetime is known)
# cluster the VM based on their lifetime into several buckets and use firstfit to allocate each bucket
import numpy as np

def clairvoyant_ltfit(state, env, lt_thre):
    obs = state["obs"].copy()
    feat = state["feat"].copy()
    numa_avail = state["avail"][1:].copy()
    lt = state["lt"]
    ctime = env.t
    # cluster the pm based on their lifetime (long or short currently!)
    # 17000 is determined by the mean of the lt for all requests
    # 1 for long and 0 for short
    request_is_split = feat[2]
    request_type = 1 if lt > lt_thre else 0

    len_buckest = 2
    pm_buckets = [[] for i in range(len_buckest)]

    for idx, pm in enumerate(env.cluster.active_pms):
        # split the pm by their type
        pm.updatetype(lt_thre, env.t)
        if np.any(numa_avail[[idx*2, idx*2+1]]):
            if pm.pm_type is None:
                # if there is one new available pm
                if lt > lt_thre:
                    pm.pm_type = 1
                else:
                    pm.pm_type = 0
            if pm.pm_type == 1:
                pm_buckets[1].append([idx, pm])
            else:
                pm_buckets[0].append([idx, pm])
    # breakpoint()
    if len(pm_buckets[request_type]) == 0:
        # no pm have the request type, create a new pm
        return 0
    else:
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
