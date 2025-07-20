''' Random '''
import numpy as np

def random_fit(state):
    pm_avail = state["avail"][1:]
    if np.any(pm_avail):
        action = np.random.choice(np.where(pm_avail)[0] + 1)
    else:
        action = 0 # Open a new PM if no PM is available!
    return action
