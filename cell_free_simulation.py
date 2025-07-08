
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

class CellFreeSimulation:
    def __init__(self, M=32, K=20, L=4, tau_p=8, pilot_reuse_dist=250, area_size=1000, RB_per_AP=4, MAX_UEs_per_RB=1, dt=1.0, seed=1):
        self.M = M
        self.K = K
        self.L = L
        self.tau_p = tau_p
        self.pilot_reuse_dist = pilot_reuse_dist
        self.area_size = area_size
        self.RB_per_AP = RB_per_AP
        self.MAX_UEs_per_RB = MAX_UEs_per_RB
        self.dt = dt
        self.seed = seed
        self.reset()

    def reset(self):
        np.random.seed(self.seed)
        self.UE_pos = np.random.uniform(0, self.area_size, (self.K, 2))
        self.AP_pos = np.random.uniform(0, self.area_size, (self.M, 2))
        self.UE_velocities = np.random.uniform(-1.5, 1.5, (self.K, 2))
        self.mobility_trace = []
        self._update_state()

    def pathloss(self, d):
        d0 = 1
        PL0_dB = -30
        alpha = 3.7
        d = np.maximum(d, d0)
        PL_dB = PL0_dB - 10 * alpha * np.log10(d / d0)
        return 10 ** (PL_dB / 10)

    def assign_pilots(self, UE_pos):
        pilot_assignments = -1 * np.ones(self.K, dtype=int)
        for k in range(self.K):
            for p in range(self.tau_p):
                conflict = False
                for j in range(k):
                    if pilot_assignments[j] == p:
                        if np.linalg.norm(UE_pos[k] - UE_pos[j]) < self.pilot_reuse_dist:
                            conflict = True
                            break
                if not conflict:
                    pilot_assignments[k] = p
                    break
            if pilot_assignments[k] == -1:
                counts = [np.sum(pilot_assignments == p) for p in range(self.tau_p)]
                pilot_assignments[k] = np.argmin(counts)
        return pilot_assignments

    def cluster_users(self, beta, pilot_assignments):
        AP_resources = [[{"UEs": set(), "pilots": set()} for _ in range(self.RB_per_AP)] for _ in range(self.M)]
        UE_clusters = [[] for _ in range(self.K)]
        UE_cluster_RB = -1 * np.ones(self.K, dtype=int)

        for k in range(self.K):
            sorted_APs = np.argsort(-beta[:, k])  # Best APs first
            found = False
            for rb in range(self.RB_per_AP):
                candidate_APs = []
                for m in sorted_APs:
                    rb_data = AP_resources[m][rb]
                    # Check pilot conflict
                    if pilot_assignments[k] in rb_data["pilots"]:
                        continue
                    # Check RB load
                    if len(rb_data["UEs"]) >= self.MAX_UEs_per_RB:
                        continue
                    candidate_APs.append(m)
                    if len(candidate_APs) >= self.L:
                        break
                if len(candidate_APs) >= 2:
                    # Assign this RB to all APs in the cluster
                    for m in candidate_APs:
                        AP_resources[m][rb]["UEs"].add(k)
                        AP_resources[m][rb]["pilots"].add(pilot_assignments[k])
                    UE_clusters[k] = candidate_APs
                    UE_cluster_RB[k] = rb
                    found = True
                    break
            if not found:
                UE_clusters[k] = []
                UE_cluster_RB[k] = -1
        return UE_clusters, UE_cluster_RB, AP_resources

    def _update_state(self):
        distances = cdist(self.AP_pos, self.UE_pos)
        beta = self.pathloss(distances)
        pilot_assignments = self.assign_pilots(self.UE_pos)
        UE_clusters, UE_cluster_RB, AP_resources = self.cluster_users(beta, pilot_assignments)
        self.state = {
            "UE_pos": self.UE_pos.copy(),
            "AP_pos": self.AP_pos.copy(),
            "pilot_assignments": pilot_assignments.copy(),
            "UE_clusters": [list(cluster) for cluster in UE_clusters],
            "UE_cluster_RB": UE_cluster_RB.copy(),
            "AP_resources": AP_resources
        }
        self.mobility_trace.append(self.state.copy())

    def step(self):
        self.UE_pos += self.UE_velocities * self.dt
        self.UE_pos = np.clip(self.UE_pos, 0, self.area_size)
        self._update_state()

    def get_state(self):
        return self.state

def plot_cluster_state(state, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = None
    AP_pos = state["AP_pos"]
    UE_pos = state["UE_pos"]
    UE_clusters = state["UE_clusters"]
    # Plot APs and enumerate
    ax.scatter(AP_pos[:, 0], AP_pos[:, 1], c='red', marker='^', label='APs')
    for i, (x, y) in enumerate(AP_pos):
        ax.text(x, y, f"AP {i}", color='red', fontsize=9, ha='right', va='bottom')
    # Plot UEs and enumerate
    ax.scatter(UE_pos[:, 0], UE_pos[:, 1], c='blue', label='UEs')
    for i, (x, y) in enumerate(UE_pos):
        ax.text(x, y, f"UE {i}", color='blue', fontsize=9, ha='left', va='top')
    # Draw connections
    for k, cluster in enumerate(UE_clusters):
        for m in cluster:
            ax.plot([UE_pos[k, 0], AP_pos[m, 0]], [UE_pos[k, 1], AP_pos[m, 1]], 'k-', alpha=0.2)
    ax.set_title("UE-AP Clustering (Current State)")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.legend()
    ax.grid(True)
    ax.set_xlim(0, AP_pos[:, 0].max() * 1.05)
    ax.set_ylim(0, AP_pos[:, 1].max() * 1.05)
    return ax
