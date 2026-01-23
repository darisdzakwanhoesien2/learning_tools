import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Greedy Exchange Trajectory Tracking",
    layout="wide"
)
st.title("🔁 Greedy Exchange Algorithm for Smooth Trajectory Tracking")

# =====================================================
# DATA (two points per frame)
# =====================================================
frames = [
    [(110,275), (185,230)],   # t = 1
    [(175,330), (264,340)],   # t = 2
    [(275,300), (350,240)],   # t = 3
    [(395,310), (450,330)],   # t = 4
    [(450,308), (490,330)]    # t = 5
]

frames = [np.array(f, dtype=float) for f in frames]

# =====================================================
# HELPERS
# =====================================================
def trajectory_cost(traj):
    """
    Smoothness cost = sum of squared step lengths
    """
    cost = 0.0
    for i in range(len(traj)-1):
        cost += np.linalg.norm(traj[i+1] - traj[i])**2
    return cost

def total_cost(traj1, traj2):
    return trajectory_cost(traj1) + trajectory_cost(traj2)

def greedy_exchange(frames):
    """
    Greedy exchange across frames:
    Try swapping assignments at each frame to reduce cost.
    """
    # Initial assignment: keep order
    traj1 = [f[0] for f in frames]
    traj2 = [f[1] for f in frames]

    improved = True

    while improved:
        improved = False

        for t in range(len(frames)):
            # Propose swap at frame t
            new_traj1 = traj1.copy()
            new_traj2 = traj2.copy()

            new_traj1[t], new_traj2[t] = new_traj2[t], new_traj1[t]

            old_cost = total_cost(traj1, traj2)
            new_cost = total_cost(new_traj1, new_traj2)

            if new_cost < old_cost:
                traj1, traj2 = new_traj1, new_traj2
                improved = True

    return np.array(traj1), np.array(traj2)

# =====================================================
# INITIAL TRAJECTORIES
# =====================================================
traj1_init = np.array([f[0] for f in frames])
traj2_init = np.array([f[1] for f in frames])

cost_init = total_cost(traj1_init, traj2_init)

# =====================================================
# RUN GREEDY EXCHANGE
# =====================================================
traj1_opt, traj2_opt = greedy_exchange(frames)
cost_opt = total_cost(traj1_opt, traj2_opt)

# =====================================================
# DISPLAY COSTS
# =====================================================
st.header("📊 Trajectory Cost Comparison")

c1, c2 = st.columns(2)

with c1:
    st.subheader("Initial Assignment")
    st.write("Total cost:", round(cost_init, 2))

with c2:
    st.subheader("After Greedy Exchange")
    st.write("Total cost:", round(cost_opt, 2))

if cost_opt < cost_init:
    st.success("✅ Greedy exchange successfully reduced trajectory cost.")
else:
    st.warning("⚠️ No improvement found.")

# =====================================================
# VISUALIZATION
# =====================================================
st.divider()
st.header("📈 Trajectory Visualization")

fig, ax = plt.subplots(figsize=(8,6))

# Plot observed points
for t, f in enumerate(frames):
    ax.scatter(f[:,0], f[:,1], s=80)
    ax.text(f[0,0]+3, f[0,1]+3, f"{t+1}", fontsize=9)
    ax.text(f[1,0]+3, f[1,1]+3, f"{t+1}", fontsize=9)

# Initial trajectories
ax.plot(traj1_init[:,0], traj1_init[:,1], "--", label="Initial Trajectory 1")
ax.plot(traj2_init[:,0], traj2_init[:,1], "--", label="Initial Trajectory 2")

# Optimized trajectories
ax.plot(traj1_opt[:,0], traj1_opt[:,1], linewidth=3, label="Optimized Trajectory 1")
ax.plot(traj2_opt[:,0], traj2_opt[:,1], linewidth=3, label="Optimized Trajectory 2")

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Greedy Exchange Trajectory Optimization")
ax.grid(True)
ax.legend()

st.pyplot(fig)

# =====================================================
# DISPLAY TRAJECTORIES
# =====================================================
st.divider()
st.header("📋 Trajectory Coordinates")

c1, c2 = st.columns(2)

with c1:
    st.subheader("Optimized Trajectory 1")
    st.write(traj1_opt)

with c2:
    st.subheader("Optimized Trajectory 2")
    st.write(traj2_opt)

# =====================================================
# EXPLANATION
# =====================================================
st.divider()
st.header("🧠 How Greedy Exchange Works")

st.markdown("""
1. Start with an initial assignment of points to trajectories.
2. At each frame, try swapping the two points between trajectories.
3. Compute the total smoothness cost.
4. If swapping reduces the cost, accept the swap.
5. Repeat until no further improvement is possible.

### ✅ Smoothness Criterion
We minimize the sum of squared motion distances:

- Shorter jumps → smoother trajectory
- Fewer zig-zags → better temporal consistency

This method is fast and works well for small tracking problems.
""")

# =====================================================
# OPTIONAL INTERACTIVE MODE
# =====================================================
st.divider()
st.header("🎛 Interactive Experiment (Optional)")

noise = st.slider("Add random noise to points", 0.0, 50.0, 0.0, 1.0)

if noise > 0:
    noisy_frames = [
        f + np.random.randn(*f.shape) * noise
        for f in frames
    ]
    traj1_n, traj2_n = greedy_exchange(noisy_frames)
    cost_n = total_cost(traj1_n, traj2_n)

    st.write("New optimized cost:", round(cost_n, 2))
else:
    st.info("Move the slider to perturb the data and re-run the algorithm.")

st.caption("🚀 Greedy exchange algorithm for trajectory smoothing and data association.")
