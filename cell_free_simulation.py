
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

# --- Simulation Parameters ---
M = 32           # Number of APs
K = 20           # Number of UEs
L = 4            # APs per UE (max cluster size)
tau_p = 8        # Number of orthogonal pilots
pilot_reuse_dist = 250  # Min distance to reuse pilot (m)
area_size = 1000        # Square area (m)
RB_per_AP = 4           # Number of RBs per AP
MAX_UEs_per_RB = 3      # Max UEs per RB per AP
time_steps = 10         # Number of time steps for mobility
dt = 1.0                # Time interval (seconds)

np.random.seed(1)

# --- UE and AP Deployment ---
UE_pos = np.random.uniform(0, area_size, (K, 2))
AP_pos = np.random.uniform(0, area_size, (M, 2))
UE_velocities = np.random.uniform(-1.5, 1.5, (K, 2))  # Random velocity (m/s)

def pathloss(d):
    d0 = 1
    PL0_dB = -30
    alpha = 3.7
    d = np.maximum(d, d0)
    PL_dB = PL0_dB - 10 * alpha * np.log10(d / d0)
    return 10 ** (PL_dB / 10)

def assign_pilots(UE_pos):
    pilot_assignments = -1 * np.ones(K, dtype=int)
    for k in range(K):
        for p in range(tau_p):
            conflict = False
            for j in range(k):
                if pilot_assignments[j] == p:
                    if np.linalg.norm(UE_pos[k] - UE_pos[j]) < pilot_reuse_dist:
                        conflict = True
                        break
            if not conflict:
                pilot_assignments[k] = p
                break
        if pilot_assignments[k] == -1:
            counts = [np.sum(pilot_assignments == p) for p in range(tau_p)]
            pilot_assignments[k] = np.argmin(counts)
    return pilot_assignments

def cluster_users(beta, pilot_assignments):
    AP_resources = [[{"UEs": set(), "pilots": set()} for _ in range(RB_per_AP)] for _ in range(M)]
    UE_clusters = [[] for _ in range(K)]
    UE_cluster_RB = -1 * np.ones(K, dtype=int)

    for k in range(K):
        sorted_APs = np.argsort(-beta[:, k])  # Best APs first
        ap_count = 0
        for m in sorted_APs:
            for rb in range(RB_per_AP):
                rb_data = AP_resources[m][rb]

                # Check pilot conflict
                if pilot_assignments[k] in rb_data["pilots"]:
                    continue
                # Check RB load
                if len(rb_data["UEs"]) >= MAX_UEs_per_RB:
                    continue
                # Ensure all APs in UE cluster share the same RB
                if len(UE_clusters[k]) > 0 and rb != UE_cluster_RB[k]:
                    continue

                # Assign
                UE_clusters[k].append(m)
                rb_data["UEs"].add(k)
                rb_data["pilots"].add(pilot_assignments[k])

                if UE_cluster_RB[k] == -1:
                    UE_cluster_RB[k] = rb

                ap_count += 1
                break  # Go to next AP

            if ap_count >= 4:
                break  # Max cluster size reached

        # If not enough APs found (less than 2), clear the cluster
        if len(UE_clusters[k]) < 2:
            for m in UE_clusters[k]:
                rb = UE_cluster_RB[k]
                AP_resources[m][rb]["UEs"].remove(k)
                AP_resources[m][rb]["pilots"].remove(pilot_assignments[k])
            UE_clusters[k] = []
            UE_cluster_RB[k] = -1

    return UE_clusters, UE_cluster_RB, AP_resources

# --- Time Evolution ---
mobility_trace = []

for t in range(time_steps):
    # Update UE positions
    UE_pos += UE_velocities * dt
    UE_pos = np.clip(UE_pos, 0, area_size)

    # Compute large-scale fading
    distances = cdist(AP_pos, UE_pos)
    beta = pathloss(distances)

    # Pilot allocation
    pilot_assignments = assign_pilots(UE_pos)

    # Clustering and resource allocation
    UE_clusters, UE_cluster_RB, AP_resources = cluster_users(beta, pilot_assignments)

    mobility_trace.append({
        "UE_pos": UE_pos.copy(),
        "pilot_assignments": pilot_assignments.copy(),
        "UE_clusters": [list(cluster) for cluster in UE_clusters],
        "UE_cluster_RB": UE_cluster_RB.copy()
    })

# --- Visualization of Last Time Step ---
plt.figure(figsize=(8, 8))
plt.scatter(AP_pos[:, 0], AP_pos[:, 1], c='red', marker='^', label='APs')
plt.scatter(UE_pos[:, 0], UE_pos[:, 1], c='blue', label='UEs')
for k, cluster in enumerate(UE_clusters):
    for m in cluster:
        plt.plot([UE_pos[k, 0], AP_pos[m, 0]], [UE_pos[k, 1], AP_pos[m, 1]], 'k-', alpha=0.2)
plt.title("UE-AP Clustering (Final Time Step)")
plt.legend()
plt.grid(True)
plt.show()
