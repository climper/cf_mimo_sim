import streamlit as st
import matplotlib.pyplot as plt
from cell_free_simulation import CellFreeSimulation, plot_cluster_state
import numpy as np

st.set_page_config(page_title="Cell-Free MIMO Simulator", layout="centered")
st.title("Cell-Free Massive MIMO Cluster Visualization")

# Sidebar for simulation parameters and highlight toggle
with st.sidebar:
    st.header("Simulation Parameters")
    mode = st.radio("Simulation Mode", ["Random Walk Mode", "Single Probe Mode"])
    M = st.number_input("Number of APs (M)", min_value=1, max_value=100, value=32)
    K = st.number_input("Number of UEs (K)", min_value=1, max_value=100, value=20)
    L = st.number_input("APs per UE (L)", min_value=1, max_value=10, value=4)
    tau_p = st.number_input("Number of Orthogonal Pilots (tau_p)", min_value=1, max_value=32, value=8)
    pilot_reuse_dist = st.number_input("Pilot Reuse Distance (m)", min_value=1, max_value=2000, value=250)
    area_size = st.number_input("Area Size (m)", min_value=100, max_value=5000, value=1000)
    RB_per_AP = st.number_input("Resource Blocks per AP", min_value=1, max_value=20, value=4)
    MAX_UEs_per_RB = st.number_input("Max UEs per RB per AP", min_value=1, max_value=10, value=1)
    dt = st.number_input("Time Step (s)", min_value=0.1, max_value=10.0, value=1.0, step=0.1, format="%.1f")
    seed = st.number_input("Random Seed", min_value=0, max_value=9999, value=1)
    st.markdown("---")
    highlight_new = st.checkbox("Highlight new connections (green)", value=False)
    reset_sim = st.button("Reset Simulation")
    probe_ue = 0
    if mode == "Single Probe Mode":
        probe_ue = st.selectbox("Select Probe UE (moving)", list(range(K)), index=0)

# Session state for simulation and history
if 'sim' not in st.session_state or reset_sim or ('mode' in st.session_state and st.session_state['mode'] != mode):
    st.session_state.sim = CellFreeSimulation(
        M=int(M), K=int(K), L=int(L), tau_p=int(tau_p), pilot_reuse_dist=pilot_reuse_dist,
        area_size=area_size, RB_per_AP=int(RB_per_AP), MAX_UEs_per_RB=int(MAX_UEs_per_RB), dt=dt, seed=int(seed)
    )
    st.session_state.step_count = 0
    st.session_state.history = [st.session_state.sim.get_state()]
    st.session_state.history_ptr = 0
    st.session_state.mode = mode
    st.session_state.probe_ue = probe_ue

sim = st.session_state.sim

col1, col2 = st.columns([3, 1])

with col1:
    st.subheader(f"Cluster Formation (Step {st.session_state.step_count})")
    # Get the state to display from history
    state = st.session_state.history[st.session_state.history_ptr]

    # Custom plotting with highlighting and probe UE
    fig, ax = plt.subplots(figsize=(7, 7))
    AP_pos = state["AP_pos"]
    UE_pos = state["UE_pos"]
    UE_clusters = state["UE_clusters"]
    # Plot APs and enumerate
    ax.scatter(AP_pos[:, 0], AP_pos[:, 1], c='red', marker='^', label='APs')
    for i, (x, y) in enumerate(AP_pos):
        ax.text(x, y, f"AP {i}", color='red', fontsize=12, ha='right', va='bottom', bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=0.5))
    # Plot UEs and enumerate
    for i, (x, y) in enumerate(UE_pos):
        if mode == "Single Probe Mode" and i == st.session_state.probe_ue:
            ax.scatter([x], [y], c='magenta', label='Probe UE' if i == 0 else None, zorder=5)
            ax.text(x, y, f"UE {i}", color='magenta', fontsize=12, ha='left', va='top', bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=0.5))
        else:
            ax.scatter([x], [y], c='blue', label='UEs' if i == 0 else None, zorder=4)
            ax.text(x, y, f"UE {i}", color='blue', fontsize=12, ha='left', va='top', bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=0.5))
    # Highlight logic
    prev_connections = set()
    if highlight_new and st.session_state.history_ptr > 0:
        prev_state = st.session_state.history[st.session_state.history_ptr - 1]
        prev_clusters = prev_state["UE_clusters"]
        for k, cluster in enumerate(prev_clusters):
            for m in cluster:
                prev_connections.add((k, m))
    # Draw connections
    for k, cluster in enumerate(UE_clusters):
        for m in cluster:
            color = 'k'
            linewidth = 1
            alpha = 0.2
            if mode == "Single Probe Mode" and k == st.session_state.probe_ue:
                if highlight_new and st.session_state.history_ptr > 0:
                    if (k, m) not in prev_connections:
                        color = 'g'
                        linewidth = 2
                        alpha = 0.7
                    else:
                        color = 'magenta'
                        linewidth = 2
                        alpha = 0.7
                else:
                    color = 'magenta'
                    linewidth = 2
                    alpha = 0.7
            elif highlight_new and st.session_state.history_ptr > 0:
                if (k, m) not in prev_connections:
                    color = 'g'
                    linewidth = 2
                    alpha = 0.7
            ax.plot([UE_pos[k, 0], AP_pos[m, 0]], [UE_pos[k, 1], AP_pos[m, 1]], color=color, alpha=alpha, linewidth=linewidth)
    ax.set_title("UE-AP Clustering (Current State)")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.legend()
    ax.grid(True)
    ax.set_xlim(0, AP_pos[:, 0].max() * 1.05)
    ax.set_ylim(0, AP_pos[:, 1].max() * 1.05)
    st.pyplot(fig)

with col2:
    st.write("")
    st.write("")
    col_prev, col_next = st.columns(2)
    with col_prev:
        if st.button("Prev Step"):
            if st.session_state.history_ptr > 0:
                st.session_state.history_ptr -= 1
                st.session_state.step_count -= 1
                st.rerun()
    with col_next:
        if st.button("Next Step"):
            # Only step if at the end of history
            if st.session_state.history_ptr == len(st.session_state.history) - 1:
                # Move logic based on mode
                if mode == "Random Walk Mode":
                    sim.UE_pos += sim.UE_velocities * sim.dt
                    sim.UE_pos = np.clip(sim.UE_pos, 0, sim.area_size)
                elif mode == "Single Probe Mode":
                    idx = st.session_state.probe_ue
                    sim.UE_pos[idx] += sim.UE_velocities[idx] * sim.dt
                    sim.UE_pos[idx] = np.clip(sim.UE_pos[idx], 0, sim.area_size)
                sim._update_state()
                st.session_state.step_count += 1
                # Store new state, keep only last 3
                st.session_state.history.append(sim.get_state())
                if len(st.session_state.history) > 3:
                    st.session_state.history.pop(0)
                else:
                    st.session_state.history_ptr += 1
            else:
                # Move forward in history if not at the end
                if st.session_state.history_ptr < len(st.session_state.history) - 1:
                    st.session_state.history_ptr += 1
                    st.session_state.step_count += 1
            st.rerun()
    st.write("")
    st.write("")
    st.markdown("**Legend:**")
    st.markdown("- <span style='color:red'>▲</span> APs", unsafe_allow_html=True)
    st.markdown("- <span style='color:blue'>●</span> UEs", unsafe_allow_html=True)
    st.markdown("- <span style='color:magenta'>●</span> Probe UE (Single Probe Mode)", unsafe_allow_html=True)
    st.markdown("- <span style='color:black'>—</span> Cluster Connections", unsafe_allow_html=True)
    st.markdown("- <span style='color:green'>—</span> New Connections (if highlighted)", unsafe_allow_html=True)

    # Monitoring dropdowns
    st.write("")
    st.write("")
    st.markdown("---")
    st.markdown("### Monitoring")
    monitor_mode = st.radio("Select entity to monitor:", ["UE", "AP"])
    if monitor_mode == "UE":
        ue_idx = st.selectbox("Select UE:", list(range(len(state["UE_clusters"]))))
        st.markdown(f"**UE {ue_idx}**")
        st.write("Serving APs:", state["UE_clusters"][ue_idx])
        st.write("Pilot Assignment:", state["pilot_assignments"][ue_idx])
        st.write("Resource Block (RB):", int(state["UE_cluster_RB"][ue_idx]) if state["UE_cluster_RB"][ue_idx] != -1 else "None")
    else:
        ap_idx = st.selectbox("Select AP:", list(range(len(state["AP_pos"]))))
        st.markdown(f"**AP {ap_idx}**")
        ap_resources = state["AP_resources"][ap_idx]
        for rb_idx, rb in enumerate(ap_resources):
            st.write(f"RB {rb_idx}: UEs = {sorted(list(rb['UEs']))}, Pilots = {sorted(list(rb['pilots']))}") 