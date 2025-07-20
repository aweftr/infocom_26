''' Select the first NUMA or server that can accommodate the VM request '''
import numpy as np

def first_fit(state):
    pm_avail = state["avail"][1:]
    if np.any(pm_avail):
        action = (np.where(pm_avail)[0] + 1)[0]
    else:
        action = 0 # Open a new PM if no PM is available!
    return action
