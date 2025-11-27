# scripts/plot_reference_policy.py
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from policies.reference_policy import GaitParams, gait_targets


def main():
    # Define gait parameters
    params = GaitParams(
        stride_length=0.04,      # meters
        stride_height=0.02,      # meters
        cycle_duration=1.5,     # seconds per full gait
        stance_fraction=0.6,
    )

    # Simulation horizon: a few cycles
    n_cycles = 3
    dt = 0.01
    T = n_cycles * params.cycle_duration
    ts = np.arange(0.0, T, dt)

    left_x, left_z = [], []
    right_x, right_z = [], []

    for t in ts:
        targets = gait_targets(t, params)
        l = targets["left"]
        r = targets["right"]

        left_x.append(l[0])
        left_z.append(l[1])

        right_x.append(r[0])
        right_z.append(r[1])

    left_x = np.array(left_x)
    left_z = np.array(left_z)
    right_x = np.array(right_x)
    right_z = np.array(right_z)

    # ----- Plot x(t) and z(t) -----
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ax = axes[0]
    ax.plot(ts, left_x, label="left x")
    ax.plot(ts, right_x, label="right x", linestyle="--")
    ax.set_ylabel("x [m]")
    ax.set_title("Foot forward/backward position over time")
    ax.grid(True)
    ax.legend()

    ax = axes[1]
    ax.plot(ts, left_z, label="left z")
    ax.plot(ts, right_z, label="right z", linestyle="--")
    ax.set_ylabel("z [m]")
    ax.set_xlabel("time [s]")
    ax.set_title("Foot height over time")
    ax.grid(True)
    ax.legend()

    fig.tight_layout()

    # ----- Plot spatial trajectory (x vs z) -----
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    ax2.plot(left_x, left_z, label="left foot")
    ax2.plot(right_x, right_z, label="right foot")
    ax2.set_xlabel("x [m]")
    ax2.set_ylabel("z [m]")
    ax2.set_title("Foot path in sagittal plane")
    ax2.grid(True)
    ax2.legend()
    ax2.axhline(0.0, color="k", linewidth=0.5)  # ground

    plt.show()


if __name__ == "__main__":
    main()
