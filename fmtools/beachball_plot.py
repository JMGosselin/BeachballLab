"""
Fuzzy beachball plot: posterior focal-mechanism uncertainty with the
observed polarities on top (lower hemisphere, equal-area projection).
"""

import numpy as np
from matplotlib.patches import Circle
from pyrocko import moment_tensor
from pyrocko.plot import beachball


def project_lower_hemisphere(azimuth_deg, takeoff_deg):
    """
    Equal-area (Lambert) projection of rays onto the lower hemisphere.
    Up-going rays (take-off > 90 deg) are plotted at the antipodal point,
    which has the same P polarity. Returns x (east), y (north).
    """
    th = np.deg2rad(np.asarray(takeoff_deg, dtype=float))
    az = np.deg2rad(np.asarray(azimuth_deg, dtype=float))
    north = np.sin(th) * np.cos(az)
    east = np.sin(th) * np.sin(az)
    down = np.cos(th)
    flip = down < 0
    north, east, down = np.where(flip, -north, north), np.where(flip, -east, east), np.abs(down)
    k = np.sqrt(1.0 / (1.0 + down))
    return east * k, north * k


def nodal_plane_xy(strike_deg, dip_deg, npts=721):
    """
    Projected trace of one nodal plane on the lower hemisphere.
    """
    s = np.deg2rad(strike_deg)
    d = np.deg2rad(dip_deg)
    strike_vec = np.array([np.cos(s), np.sin(s), 0.0])
    dip_vec = np.array([-np.sin(s) * np.cos(d), np.cos(s) * np.cos(d), np.sin(d)])
    t = np.linspace(0.0, np.pi, npts)
    v = np.outer(np.cos(t), strike_vec) + np.outer(np.sin(t), dip_vec)
    north, east, down = v[:, 0], v[:, 1], np.clip(v[:, 2], 0.0, None)
    k = np.sqrt(1.0 / (1.0 + down))
    return east * k, north * k


def plot_fuzzy_beachball(cfg, event_row, result, sdr, solution_label, filename):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_cfg = cfg.get("plotting", {}) or {}
    n_draws = int(plot_cfg.get("n_posterior_draws", 500))
    rng = np.random.default_rng(int(plot_cfg.get("random_seed", 1234)))

    grid = result["grid"]
    posterior = result["posterior"]
    obs = result["obs"]

    draw_idx = rng.choice(len(grid), size=n_draws, replace=True, p=posterior)
    mts = [
        moment_tensor.MomentTensor(strike=s, dip=d, rake=r, scalar_moment=1.0).m6()
        for s, d, r in grid[draw_idx]
    ]

    fig, ax = plt.subplots(figsize=(8, 8.6))

    beachball.plot_fuzzy_beachball_mpl_pixmap(
        mts,
        ax,
        position=(0.0, 0.0),
        size=2.0,
        size_units="data",
        projection="lambert",
        grid_resolution=int(plot_cfg.get("fuzzy_grid_resolution", 300)),
        zorder=1,
    )

    ax.add_patch(Circle((0, 0), 1.0, facecolor="none", edgecolor="black", lw=1.5, zorder=10))

    mt_pref = moment_tensor.MomentTensor(strike=sdr[0], dip=sdr[1], rake=sdr[2])
    for plane in mt_pref.both_strike_dip_rake():
        x, y = nodal_plane_xy(plane[0], plane[1])
        ax.plot(x, y, color="black", lw=2.0, zorder=12)

    for text, xy, ha, va in [("N", (0, 1.05), "center", "bottom"), ("E", (1.05, 0), "left", "center"),
                             ("S", (0, -1.05), "center", "top"), ("W", (-1.05, 0), "right", "center")]:
        ax.text(*xy, text, ha=ha, va=va, fontsize=12)

    x, y = project_lower_hemisphere(obs["azimuth"], obs["takeoff_angle"])
    conf = obs["confidence"].to_numpy(dtype=float)
    sizes = 40.0 + 120.0 * np.clip(np.nan_to_num(conf, nan=0.5), 0.0, 1.0)
    up = (obs["polarity"] == "U").to_numpy()
    down = ~up

    ax.scatter(x[up], y[up], s=sizes[up], marker="^", c="tab:blue", edgecolors="black", label="Up (compression)", zorder=30)
    ax.scatter(x[down], y[down], s=sizes[down], marker="v", c="white", edgecolors="black", label="Down (dilatation)", zorder=30)

    if plot_cfg.get("show_station_labels", True):
        for xi, yi, sta in zip(x, y, obs["sta"]):
            ax.annotate(sta, (xi, yi), xytext=(4, 4), textcoords="offset points", fontsize=7, zorder=40,
                        bbox=dict(boxstyle="round,pad=0.1", facecolor=(1, 1, 1, 0.7), edgecolor="none"))

    plane1, plane2 = mt_pref.both_strike_dip_rake()
    ax.set_title(
        "Event {}  M{}  depth {:.1f} km\n"
        "{} solution: strike/dip/rake {:.0f}/{:.0f}/{:.0f} and {:.0f}/{:.0f}/{:.0f}\n"
        "{} polarities, {} posterior draws, model {}".format(
            event_row["event_id"], event_row["event_mag"], event_row["event_depth"],
            solution_label.capitalize(), *plane1, *plane2,
            len(obs), n_draws, cfg["velocity_model"]["model"]),
        fontsize=10,
    )
    ax.legend(loc="lower left", bbox_to_anchor=(-0.02, -0.06), fontsize=9, framealpha=0.9)
    ax.set_aspect("equal")
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.axis("off")

    fig.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close(fig)
