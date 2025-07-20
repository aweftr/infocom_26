''' Select the NUMA or server where the last recently installed VM was placed '''
import numpy as np

def movetofront_fit(state, cluster):
    avails = state["avail"][1:]
    recent_time_avail = {}
    for idx, avail in enumerate(avails):
        if avail:
            recent_time_avail[idx] = cluster.active_pms[idx // 2].recent_usage_time
    if len(recent_time_avail) == 0:
        return 0
    else:
        action = sorted(recent_time_avail.items(), key=lambda x: x[1])[0][0] + 1
        return action
