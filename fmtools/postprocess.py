"""
Post-processing of the focal-mechanism grid search: principal stress-axis
orientations, mechanism uncertainty and per-station polarity predictions.

Uncertainty is described with the rotation between each posterior sample
and the preferred mechanism. Every double couple can be turned into any
other by a single 3-D rotation; its angle (the Kagan angle, 0-120 deg) is
how different the two mechanisms are. From the rotation vectors of many
posterior samples we compute

    C = mean( r r^T )        (rotation-vector dispersion matrix, RVDM, deg^2)

whose square-rooted trace (sigma) is an overall uncertainty and whose
eigenvectors show the rotation axes about which the mechanism is least
(first) and best (last) constrained. sigma_iso is the value for a
completely unconstrained (uniformly random) mechanism.
"""

import json

import numpy as np
import pandas as pd

from .inversion import p_amplitudes, ray_vectors, sdr_to_moment_tensor


# Symmetry operations of a double couple in its (T, N, P) frame.
DC_SYMMETRY = np.array([
    np.diag([1.0, 1.0, 1.0]),
    np.diag([1.0, -1.0, -1.0]),
    np.diag([-1.0, 1.0, -1.0]),
    np.diag([-1.0, -1.0, 1.0]),
])

SIGMA_ISO_DEG = float(np.degrees(np.pi * np.sqrt(3.0) / 4.0))


# ----------------------------------------------------------------------
# Reading inversion results
# ----------------------------------------------------------------------
def load_inversion(npz_path):
    npz = np.load(str(npz_path), allow_pickle=False)
    meta = json.loads(str(npz["metadata_json"]))
    obs = pd.DataFrame(
        {
            "net": npz["obs_network"],
            "sta": npz["obs_station"],
            "chan": npz["obs_channel"],
            "polarity": npz["obs_polarity"],
            "confidence": npz["obs_confidence"],
            "azimuth": npz["azimuth_deg"],
            "takeoff_angle": npz["takeoff_deg"],
        }
    )
    grid = np.column_stack([npz["strike"], npz["dip"], npz["rake"]])
    return {
        "meta": meta,
        "grid": grid,
        "posterior": np.asarray(npz["posterior"], dtype=float),
        "best_sdr": np.asarray(npz["best_sdr"], dtype=float),
        "centroid_sdr": np.asarray(npz["centroid_sdr"], dtype=float),
        "preferred_sdr": np.asarray(meta["preferred_sdr"], dtype=float),
        "preferred_solution": meta["preferred_solution"],
        "obs": obs,
    }


# ----------------------------------------------------------------------
# Principal axes
# ----------------------------------------------------------------------
def tnp_frames(m):
    """
    Right-handed (T, N, P) eigenvector frames for moment tensors of shape
    (..., 3, 3) in NED. Returns (..., 3, 3) with columns T, N, P.
    """
    _, vecs = np.linalg.eigh(m)  # ascending eigenvalues: P, N, T
    frames = np.stack([vecs[..., :, 2], vecs[..., :, 1], vecs[..., :, 0]], axis=-1)
    flip = np.linalg.det(frames) < 0
    frames[flip, :, 1] *= -1.0
    return frames


def trend_plunge(v_ned):
    """
    Trend (deg from north) and plunge (deg, positive down) of axes given as
    NED vectors (..., 3). Axes are lines, so the downward end is used.
    """
    v = np.asarray(v_ned, dtype=float)
    v = np.where(v[..., 2:3] < 0, -v, v)
    trend = np.degrees(np.arctan2(v[..., 1], v[..., 0])) % 360.0
    plunge = np.degrees(np.arctan2(v[..., 2], np.hypot(v[..., 0], v[..., 1])))
    return trend, plunge


def faulting_regime(p_plunge, t_plunge, n_plunge):
    """
    Faulting style from the plunges of P, T and N (B) axes, following the
    World Stress Map scheme (Zoback, 1992):
        NF normal, NS normal/strike-slip, SS strike-slip,
        TS thrust/strike-slip, TF thrust, U unclassified (oblique).
    Works element-wise on arrays.
    """
    pp, pt, pn = np.broadcast_arrays(
        np.asarray(p_plunge, float), np.asarray(t_plunge, float), np.asarray(n_plunge, float))
    regime = np.full(pp.shape, "U", dtype="<U2")
    regime[(pp >= 52) & (pt <= 35)] = "NF"
    regime[(pp >= 40) & (pp < 52) & (pt <= 20)] = "NS"
    regime[((pp < 40) & (pn >= 45) & (pt <= 20)) | ((pp <= 20) & (pn >= 45) & (pt < 40))] = "SS"
    regime[(pp <= 20) & (pt >= 40) & (pt < 52)] = "TS"
    regime[(pp <= 35) & (pt >= 52)] = "TF"
    return regime


REGIME_NAMES = {
    "NF": "normal",
    "NS": "normal / strike-slip",
    "SS": "strike-slip",
    "TS": "thrust / strike-slip",
    "TF": "thrust",
    "U": "oblique (unclassified)",
}


# ----------------------------------------------------------------------
# Uncertainty
# ----------------------------------------------------------------------
def rotation_vectors(frame_ref, frames):
    """
    Smallest rotation (as rotation vectors in degrees, NED) taking the
    reference double couple onto each of ``frames`` (n, 3, 3), taking the
    four-fold double-couple symmetry into account. Its length is the
    Kagan angle.
    """
    best_angle = np.full(len(frames), np.inf)
    best_r = np.zeros((len(frames), 3))
    for S in DC_SYMMETRY:
        R = frames @ S @ frame_ref.T
        cos_a = np.clip((np.trace(R, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
        angle = np.arccos(cos_a)
        axis = np.stack([R[:, 2, 1] - R[:, 1, 2], R[:, 0, 2] - R[:, 2, 0], R[:, 1, 0] - R[:, 0, 1]], axis=1)
        norm = np.linalg.norm(axis, axis=1)
        axis = np.divide(axis, norm[:, None], out=np.zeros_like(axis), where=norm[:, None] > 1e-12)
        better = angle < best_angle
        best_angle[better] = angle[better]
        best_r[better] = np.degrees(angle[better])[:, None] * axis[better]
    return best_r


def mechanism_uncertainty(inv, n_samples, rng):
    """
    Draw posterior samples and describe their spread around the preferred
    mechanism.
    """
    grid = inv["grid"]
    idx = rng.choice(len(grid), size=int(n_samples), replace=True, p=inv["posterior"])
    samples = grid[idx]

    frame_ref = tnp_frames(sdr_to_moment_tensor(*inv["preferred_sdr"]))
    frames = tnp_frames(sdr_to_moment_tensor(samples[:, 0], samples[:, 1], samples[:, 2]))

    rv = rotation_vectors(frame_ref, frames)
    kagan = np.linalg.norm(rv, axis=1)
    C = rv.T @ rv / len(rv)
    sigma = float(np.sqrt(np.trace(C)))

    evals, evecs = np.linalg.eigh(C)
    order = np.argsort(evals)[::-1]

    t_tr, t_pl = trend_plunge(frames[:, :, 0])
    n_tr, n_pl = trend_plunge(frames[:, :, 1])
    p_tr, p_pl = trend_plunge(frames[:, :, 2])

    return {
        "samples": samples,
        "kagan": kagan,
        "rvdm": C,
        "sigma": sigma,
        "sigma_eff": float(sigma / (SIGMA_ISO_DEG - sigma)) if sigma < SIGMA_ISO_DEG else np.inf,
        "rvdm_ratio": float(np.trace(C) / SIGMA_ISO_DEG**2),
        "rvdm_eigenvalues": evals[order],
        "rvdm_axes": [trend_plunge(evecs[:, i]) for i in order],
        "axes": {"P": (p_tr, p_pl), "T": (t_tr, t_pl), "N": (n_tr, n_pl)},
        "regimes": faulting_regime(p_pl, t_pl, n_pl),
    }


def preferred_axes(sdr):
    frame = tnp_frames(sdr_to_moment_tensor(*sdr))
    return {name: trend_plunge(frame[:, i]) for i, name in enumerate("TNP")}


# ----------------------------------------------------------------------
# Station predictions
# ----------------------------------------------------------------------
def station_predictions(inv):
    """
    For every observed polarity: the polarity and amplitude predicted by
    the preferred mechanism, and the posterior probability that the first
    motion is UP (averaged over all mechanisms, weighted by the posterior).
    """
    obs = inv["obs"].copy()
    rays = ray_vectors(obs["azimuth"], obs["takeoff_angle"])

    amp_pref = p_amplitudes(sdr_to_moment_tensor(*inv["preferred_sdr"])[None], rays)[0]

    grid = inv["grid"]
    keep = inv["posterior"] > 1e-12 * inv["posterior"].max()
    m = sdr_to_moment_tensor(grid[keep, 0], grid[keep, 1], grid[keep, 2])
    w = inv["posterior"][keep] / inv["posterior"][keep].sum()
    prob_up = (w[:, None] * (p_amplitudes(m, rays) > 0)).sum(axis=0)

    obs["predicted"] = np.where(amp_pref >= 0, "U", "D")
    obs["amplitude"] = amp_pref.round(3)
    obs["prob_up"] = prob_up.round(3)
    obs["prob_observed"] = np.where(obs["polarity"] == "U", prob_up, 1.0 - prob_up).round(3)
    obs["match"] = obs["predicted"] == obs["polarity"]
    return obs


# ----------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------
def summary_rows(inv, unc, pred):
    """
    Machine-readable (parameter_name, value) rows.
    """
    meta = inv["meta"]
    pref = inv["preferred_sdr"]
    ax = preferred_axes(pref)
    regime = faulting_regime(ax["P"][1], ax["T"][1], ax["N"][1]).item()
    p68, p95 = np.percentile(unc["kagan"], [68, 95])

    rows = [
        ("event_id", meta["event_id"]),
        ("origin_time", meta["origin_time"]),
        ("event_latitude_deg", meta["event_lat"]),
        ("event_longitude_deg", meta["event_lon"]),
        ("event_depth_km", meta["event_depth_km"]),
        ("event_magnitude", meta["magnitude"]),
        ("velocity_model", meta["velocity_model"]),
        ("number_of_polarities", meta["nobs"]),
        ("number_of_grid_points", meta["ngrid"]),
        ("preferred_solution", inv["preferred_solution"]),
        ("preferred_strike_deg", pref[0]),
        ("preferred_dip_deg", pref[1]),
        ("preferred_rake_deg", pref[2]),
        ("best_grid_strike_deg", inv["best_sdr"][0]),
        ("best_grid_dip_deg", inv["best_sdr"][1]),
        ("best_grid_rake_deg", inv["best_sdr"][2]),
        ("centroid_strike_deg", inv["centroid_sdr"][0]),
        ("centroid_dip_deg", inv["centroid_sdr"][1]),
        ("centroid_rake_deg", inv["centroid_sdr"][2]),
    ]
    for name in "PTN":
        rows += [("{}_axis_trend_deg".format(name), ax[name][0]), ("{}_axis_plunge_deg".format(name), ax[name][1])]
    rows += [
        ("faulting_regime_wsm", regime),
        ("n_posterior_samples", len(unc["kagan"])),
        ("kagan_angle_median_deg", np.median(unc["kagan"])),
        ("kagan_angle_68pct_deg", p68),
        ("kagan_angle_95pct_deg", p95),
        ("sigma_overall_deg", unc["sigma"]),
        ("sigma_isotropic_deg", SIGMA_ISO_DEG),
        ("sigma_eff_unscaled", unc["sigma_eff"]),
        ("rvdm_trace_over_sigma_iso_squared", unc["rvdm_ratio"]),
    ]
    C = unc["rvdm"]
    for i in range(3):
        for j in range(3):
            rows.append(("RVDM_C{}{}_deg2".format(i + 1, j + 1), C[i, j]))
    for i in range(3):
        rows += [
            ("RVDM_eigenvalue_{}_deg2".format(i + 1), unc["rvdm_eigenvalues"][i]),
            ("RVDM_axis_{}_trend_deg".format(i + 1), unc["rvdm_axes"][i][0]),
            ("RVDM_axis_{}_plunge_deg".format(i + 1), unc["rvdm_axes"][i][1]),
        ]
    for code in REGIME_NAMES:
        rows.append(("posterior_fraction_regime_{}".format(code), float(np.mean(unc["regimes"] == code))))
    rows += [
        ("polarities_matching_preferred", int(pred["match"].sum())),
        ("fraction_matching_preferred", float(pred["match"].mean())),
    ]
    return [(k, float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in rows]


def report_text(inv, unc, pred):
    meta = inv["meta"]
    pref = inv["preferred_sdr"]
    ax = preferred_axes(pref)
    regime = faulting_regime(ax["P"][1], ax["T"][1], ax["N"][1]).item()
    k50, k68, k95 = np.percentile(unc["kagan"], [50, 68, 95])

    lines = [
        "Focal-mechanism post-processing",
        "===============================",
        "Event {}  {}  M{}  depth {:.1f} km".format(
            meta["event_id"], meta["origin_time"], meta["magnitude"], meta["event_depth_km"]),
        "",
        "Preferred solution ({}): strike {:.0f}, dip {:.0f}, rake {:.0f}".format(inv["preferred_solution"], *pref),
        "",
        "Principal axes (trend / plunge, deg):",
        "    P (compression) {:6.0f} / {:3.0f}".format(*ax["P"]),
        "    T (tension)     {:6.0f} / {:3.0f}".format(*ax["T"]),
        "    N (null)        {:6.0f} / {:3.0f}".format(*ax["N"]),
        "Faulting style (World Stress Map): {} ({})".format(regime, REGIME_NAMES[regime]),
        "",
        "Uncertainty from {} posterior samples:".format(len(unc["kagan"])),
        "    rotation (Kagan) angle to preferred solution: median {:.0f} deg, "
        "68% within {:.0f} deg, 95% within {:.0f} deg".format(k50, k68, k95),
        "    sigma = {:.1f} deg (a random mechanism gives {:.1f} deg)".format(unc["sigma"], SIGMA_ISO_DEG),
        "",
        "Faulting style of the posterior samples:",
    ]
    for code, name in REGIME_NAMES.items():
        frac = np.mean(unc["regimes"] == code)
        if frac > 0:
            lines.append("    {:<3s} {:<24s} {:5.1f} %".format(code, name, 100 * frac))

    lines += [
        "",
        "Station predictions (prob_obs = posterior probability of the observed polarity):",
        "{:<5s} {:<6s} {:>8s} {:>8s} {:>4s} {:>5s} {:>7s} {:>8s}".format(
            "net", "sta", "azimuth", "takeoff", "obs", "pred", "amp", "prob_obs"),
    ]
    for _, r in pred.iterrows():
        flag = ""
        if not r["match"]:
            flag = "  <-- misfit"
        elif abs(r["amplitude"]) < 0.2:
            flag = "  (near nodal plane)"
        lines.append("{:<5s} {:<6s} {:8.1f} {:8.1f} {:>4s} {:>5s} {:7.2f} {:8.2f}{}".format(
            r["net"], r["sta"], r["azimuth"], r["takeoff_angle"], r["polarity"],
            r["predicted"], r["amplitude"], r["prob_observed"], flag))
    lines.append("")
    lines.append("{} of {} polarities match the preferred solution.".format(int(pred["match"].sum()), len(pred)))
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------
def plot_uncertainty(inv, unc, pred, filename, n_axes_plot=500):
    """
    Left: P, T and N axes of posterior samples on a lower-hemisphere
    stereonet, with the preferred solution. Right: histogram of rotation
    angles between posterior samples and the preferred solution.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from pyrocko import moment_tensor

    from .beachball_plot import nodal_plane_xy, project_lower_hemisphere

    meta = inv["meta"]
    pref = inv["preferred_sdr"]
    ax_pref = preferred_axes(pref)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6.2), gridspec_kw={"width_ratios": [1, 1.1]})

    ax1.add_patch(Circle((0, 0), 1.0, facecolor="none", edgecolor="black", lw=1.5))
    for plane in moment_tensor.MomentTensor(strike=pref[0], dip=pref[1], rake=pref[2]).both_strike_dip_rake():
        x, y = nodal_plane_xy(plane[0], plane[1])
        ax1.plot(x, y, color="0.3", lw=1.2)

    n = min(n_axes_plot, len(unc["kagan"]))
    styles = {"P": ("tab:blue", "P (compression)"), "T": ("tab:red", "T (tension)"), "N": ("0.5", "N (null)")}
    for name, (color, label) in styles.items():
        trend, plunge = unc["axes"][name]
        x, y = project_lower_hemisphere(trend[:n], 90.0 - plunge[:n])
        ax1.scatter(x, y, s=6, color=color, alpha=0.35, lw=0, label=label + " samples")
        xp, yp = project_lower_hemisphere(ax_pref[name][0], 90.0 - ax_pref[name][1])
        ax1.scatter(xp, yp, s=160, color=color, edgecolors="black", lw=1.2, marker="o", zorder=5)
        ax1.annotate(name, (xp, yp), xytext=(7, 7), textcoords="offset points", fontsize=12, weight="bold")

    for text, xy, ha, va in [("N", (0, 1.05), "center", "bottom"), ("E", (1.05, 0), "left", "center"),
                             ("S", (0, -1.05), "center", "top"), ("W", (-1.05, 0), "right", "center")]:
        ax1.text(*xy, text, ha=ha, va=va, fontsize=11)
    ax1.set_aspect("equal")
    ax1.set_xlim(-1.15, 1.15)
    ax1.set_ylim(-1.15, 1.15)
    ax1.axis("off")
    ax1.legend(loc="lower left", bbox_to_anchor=(-0.05, -0.08), fontsize=8, markerscale=3)
    ax1.set_title("Principal axes of {} posterior samples\n(large symbols: preferred solution)".format(n), fontsize=10)

    k50, k68, k95 = np.percentile(unc["kagan"], [50, 68, 95])
    ax2.hist(unc["kagan"], bins=np.arange(0, 122, 3), color="0.6", edgecolor="white")
    for val, lab, ls in [(k50, "median", "-"), (k68, "68%", "--"), (k95, "95%", ":")]:
        ax2.axvline(val, color="black", ls=ls, lw=1.2, label="{} {:.0f}°".format(lab, val))
    ax2.set_xlim(0, 120)
    ax2.set_xlabel("Rotation (Kagan) angle to preferred solution (deg)")
    ax2.set_ylabel("Number of posterior samples")
    ax2.legend(fontsize=9)
    ax2.set_title("Mechanism uncertainty  (sigma = {:.1f}°)".format(unc["sigma"]), fontsize=10)

    fig.suptitle(
        "Event {}  M{}   preferred {}: {:.0f}/{:.0f}/{:.0f}   {} of {} polarities fit".format(
            meta["event_id"], meta["magnitude"], inv["preferred_solution"], *pref,
            int(pred["match"].sum()), len(pred)),
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close(fig)
