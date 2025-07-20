''' Select the NUMA/server with the highest utilization for placing small/large VMs.
    Utilization is determined by the norm of the usage vector (Maximum, sum, Lp norm).
    SUM is used in this file.'''
import numpy as np

# Select the NUMA/server with the highest utilization

def best_fit(state):
    obs = state["obs"].copy()
    feat = state["feat"].copy()
    numa_avail = state["avail"][1:].copy()
    total_cpu = best_fit.total_cpu
    total_mem = best_fit.total_mem

    server_num = obs.shape[0] // 2
    # Check if the VM request is split across NUMAs
    request_is_split = feat[2]
    
    
    # Normalize the remaining resources for CPU and memory (scale `obs` by total_cpu and total_mem)
    obs[:, 0] = obs[:, 0].astype(float) / total_cpu
    obs[:, 1] = obs[:, 1].astype(float) / total_mem
    obs[numa_avail == 0] = 2
    if np.any(numa_avail):
        if not request_is_split:
            obs_sum = obs.sum(axis=1)
            action = np.argmin(obs_sum) + 1
        else:
            obs_sum = obs.reshape(-1, 4).sum(axis=1)
            action = 2 * (np.argmin(obs_sum) + 1)
    else:
        action = 0
    # # Ignore infeasible NUMAs (set the corresponding third dimension in `obs` to a large value based on `avail`)
    # obs[numa_avail == 0] = 2

    # # Compute the remaining resource rate for each NUMA (minimum of CPU and memory resources per NUMA)
    # # Reshape `obs` to (n, 2) and take the smaller value
    # obs_min = np.min(obs, axis=2)

    # if request_is_split:
    #     # Compute the remaining resource rate for each server (average of the two NUMAs)
    #     # Find the server with the smallest remaining resources (sum the rows and find the index of the minimum value)
    #     row_sums = np.sum(obs_min, axis=1)  # For comparison, summing is equivalent to averaging
    #     action = 2 * np.argmin(row_sums)
    # else:
    #     # Find the NUMA with the smallest remaining resources (locate the index of the minimum value in `obs_min`)
    #     action = np.argmin(obs_min.ravel())  # Use ravel() to flatten `obs_min` into a 1D array

    return action

