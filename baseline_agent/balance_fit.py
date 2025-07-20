''' Large VMs use the First Fit strategy, while small VMs are placed in the NUMA with the largest utilization difference (if utilization is equal, fall back to First Fit). '''
import numpy as np

def balance_fit(state, balance_type):
    """balance_type: max, sum"""
    obs = state["obs"].copy()
    feat = state["feat"].copy()
    numa_avail = state["avail"][1:].copy()
    total_cpu = balance_fit.total_cpu
    total_mem = balance_fit.total_mem
    server_num = obs.shape[0] // 2
    # Check if the VM request is split across NUMAs
    request_is_split = feat[2]
    # Normalize the remaining resources for CPU and memory (scale `obs` by total_cpu and total_mem)
    obs[:, 0] = obs[:, 0].astype(float) / total_cpu
    obs[:, 1] = obs[:, 1].astype(float) / total_mem
    obs = obs.reshape(-1, 2, 2)
    if balance_type == "max":
        # usage is max, remain usage is min
        numa_remain_resource = np.min(obs, axis=2)
    elif balance_type == "sum":
        numa_remain_resource = np.sum(obs, axis=2)

    if not np.any(numa_avail):
        return 0
    
    if request_is_split:
        # First Fit strategy
        action = np.where(numa_avail)[0][0] + 1
    else:
        server_avail = np.max(numa_avail.reshape(-1, 2), axis=1)
        differences = np.abs(numa_remain_resource[:, 0] - numa_remain_resource[:, 1]) * server_avail
        if np.all(differences == 0):
            action = np.where(numa_avail)[0][0] + 1
        else:
            # breakpoint()
            server_max_diff = np.argmax(differences)  # Find the server with the largest resource difference
            numa_action = np.argmax(numa_remain_resource[server_max_diff])  # Select the NUMA with more resources in that server
            action = (2 * server_max_diff) + numa_action + 1
            if numa_avail[action - 1] == 0:  # It is possible that the NUMA with fewer resources is available, but the NUMA with more resources is not
                # In such cases, switch the action to the other NUMA
                numa_action = 1 - numa_action  # Switch to the other NUMA
                action = (2 * server_max_diff) + numa_action + 1

    return action
