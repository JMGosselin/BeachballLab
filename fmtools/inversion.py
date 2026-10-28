"""
Probabilistic double-couple focal-mechanism inversion from P-wave
first-motion polarities (grid search over strike, dip and rake).

For every trial mechanism m and station i the predicted P amplitude A_i(m)
is computed from the ray direction (azimuth, take-off angle). The
likelihood of the observed polarity y_i (+1 up, -1 down) is

    L_i = (1 - omega) * Phi( y_i A_i / sigma_i ) + omega * Phi( -y_i A_i / sigma_i )

where Phi is the standard normal CDF, sigma_i grows as the pick confidence
decreases, and omega is the probability that a polarity is simply wrong
(e.g. a reversed station). The posterior over the grid is proportional to
the product of the L_i (uniform prior: strike and rake uniform, dip uniform
in cos(dip)).
"""

import json

import numpy as np
import pandas as pd
from scipy.special import erf
from pyrocko import moment_tensor

from .velocity import first_arrival, load_velocity_model, phase_defs


# ----------------------------------------------------------------------
# Input table: picks + station geometry + take-off angles
# ----------------------------------------------------------------------
def build_polarity_table(cfg, event_row, picks, stations):
    """
    Combine polarity picks with the station table and compute take-off
    angles with the velocity model. Returns one row per pick in the
    event-CSV format used by the inversion:

        event_id, event_datetime, event_lat, event_lon, event_depth,
        event_mag, sta, net, loc, chan, polarity, confidence, kind,
        accepted, azimuth, takeoff_angle, distance_km, station_lat,
        station_lon
    """
    inv_cfg = cfg["inversion"]

    df = picks.merge(
        stations[["network", "station", "location", "station_lat", "station_lon", "distance_km", "azimuth"]],
        on=["network", "station", "location"],
        how="left",
    )

    missing = df["distance_km"].isna()
    if missing.any():
        names = df.loc[missing, "network"] + "." + df.loc[missing, "station"]
        print("    WARNING: no station coordinates for {}; these picks are ignored.".format(", ".join(names)))
        df = df[~missing].copy()

    unknown = df["confidence"].isna()
    if unknown.any():
        kinds = sorted(set(df.loc[unknown, "kind"].astype(int)))
        print("    WARNING: marker kind(s) {} not in picking.confidence_by_kind; using confidence 0.5.".format(kinds))
        df.loc[unknown, "confidence"] = 0.5

    mod = load_velocity_model(cfg)
    phases = phase_defs(inv_cfg.get("takeoff_phases", "P,p"))
    depth_km = float(event_row["event_depth"])

    takeoff = []
    for dist in df["distance_km"]:
        arrival = first_arrival(mod, float(dist), depth_km, phases)
        takeoff.append(arrival.takeoff_angle() if arrival is not None else np.nan)
    df["takeoff_angle"] = takeoff

    # Only clear up/down polarities are used.
    df["accepted"] = (df["polarity"].isin(["U", "D"]) & np.isfinite(df["takeoff_angle"])).astype(int)

    out = pd.DataFrame(
        {
            "event_id": event_row["event_id"],
            "event_datetime": event_row["event_datetime"],
            "event_lat": event_row["event_lat"],
            "event_lon": event_row["event_lon"],
            "event_depth": event_row["event_depth"],
            "event_mag": event_row["event_mag"],
            "sta": df["station"],
            "net": df["network"],
            "loc": df["location"],
            "chan": df["channel"],
            "polarity": df["polarity"],
            "confidence": df["confidence"],
            "kind": df["kind"],
            "accepted": df["accepted"],
            "azimuth": df["azimuth"].round(2),
            "takeoff_angle": df["takeoff_angle"].round(2),
            "distance_km": df["distance_km"].round(2),
            "station_lat": df["station_lat"],
            "station_lon": df["station_lon"],
        }
    )
    return out.sort_values("distance_km").reset_index(drop=True)


# ----------------------------------------------------------------------
# Forward model
# ----------------------------------------------------------------------
def sdr_to_moment_tensor(strike, dip, rake):
    """
    Double-couple moment tensors (unit scalar moment, NED coordinates,
    Aki & Richards convention) for arrays of strike, dip, rake in degrees.
    Returns an array of shape (..., 3, 3). Identical to
    pyrocko.moment_tensor.MomentTensor(strike, dip, rake).m().
    """
    s, d, r = [np.deg2rad(np.asarray(x, dtype=float)) for x in (strike, dip, rake)]
    ss, cs = np.sin(s), np.cos(s)
    sd, cd = np.sin(d), np.cos(d)
    sr, cr = np.sin(r), np.cos(r)
    s2s, c2s = np.sin(2 * s), np.cos(2 * s)
    s2d, c2d = np.sin(2 * d), np.cos(2 * d)

    m = np.empty(np.shape(s) + (3, 3))
    m[..., 0, 0] = -(sd * cr * s2s + s2d * sr * ss**2)
    m[..., 1, 1] = sd * cr * s2s - s2d * sr * cs**2
    m[..., 2, 2] = s2d * sr
    m[..., 0, 1] = m[..., 1, 0] = sd * cr * c2s + 0.5 * s2d * sr * s2s
    m[..., 0, 2] = m[..., 2, 0] = -(cd * cr * cs + c2d * sr * ss)
    m[..., 1, 2] = m[..., 2, 1] = -(cd * cr * ss - c2d * sr * cs)
    return m


def ray_vectors(azimuth_deg, takeoff_deg):
    """
    Unit ray directions (north, east, down). Take-off angle is measured
    from the downward vertical, azimuth clockwise from north.
    """
    th = np.deg2rad(np.asarray(takeoff_deg, dtype=float))
    az = np.deg2rad(np.asarray(azimuth_deg, dtype=float))
    return np.stack([np.sin(th) * np.cos(az), np.sin(th) * np.sin(az), np.cos(th)], axis=-1)


def p_amplitudes(m, rays):
    """
    Radiated P amplitude r^T M r for every mechanism (N) and ray (S):
    positive = compression = first motion UP. Shape (N, S).
    """
    return np.einsum("si,nij,sj->ns", rays, m, rays)


# ----------------------------------------------------------------------
# Grid search
# ----------------------------------------------------------------------
def sdr_grid(inv_cfg):
    """
    Strike/dip/rake grid. Dip is sampled uniformly in cos(dip), which gives
    a uniform prior over fault-plane orientations.
    """
    strikes = np.arange(0.0, 360.0, float(inv_cfg["strike_step"]))
    rakes = np.arange(-180.0, 180.0, float(inv_cfg["rake_step"]))
    cos_dips = np.linspace(
        np.cos(np.deg2rad(float(inv_cfg["dip_max"]))),
        np.cos(np.deg2rad(float(inv_cfg["dip_min"]))),
        int(inv_cfg["n_dips"]),
    )
    dips = np.rad2deg(np.arccos(cos_dips))
    S, D, R = np.meshgrid(strikes, dips, rakes, indexing="ij")
    return np.column_stack([S.ravel(), D.ravel(), R.ravel()])


def invert_polarities(cfg, table):
    """
    Run the grid search for one event. ``table`` is the output of
    build_polarity_table. Returns a dict with the grid, posterior and
    preferred solutions.
    """
    inv_cfg = cfg["inversion"]
    obs = table[table["accepted"] == 1].reset_index(drop=True)

    yi = np.where(obs["polarity"] == "U", 1.0, -1.0)
    conf = np.clip(obs["confidence"].to_numpy(dtype=float), 0.0, 1.0)
    conf[~np.isfinite(conf)] = 0.5
    sigma_min = float(inv_cfg["sigma_min"])
    sigma_max = float(inv_cfg["sigma_max"])
    sigma_i = sigma_min + (1.0 - conf) * (sigma_max - sigma_min)
    omega = float(inv_cfg["omega"])

    grid = sdr_grid(inv_cfg)
    m = sdr_to_moment_tensor(grid[:, 0], grid[:, 1], grid[:, 2])
    amps = p_amplitudes(m, ray_vectors(obs["azimuth"], obs["takeoff_angle"]))

    arg = yi * amps / (np.sqrt(2.0) * sigma_i)
    like_i = 0.5 * (1.0 + erf(arg)) * (1.0 - omega) + 0.5 * (1.0 - erf(arg)) * omega
    log_like = np.sum(np.log(np.clip(like_i, 1e-300, 1.0)), axis=1)

    posterior = np.exp(log_like - log_like.max())
    posterior /= posterior.sum()

    best_idx = int(np.argmax(posterior))
    best_sdr = grid[best_idx].copy()

    # Posterior-weighted mean moment tensor, reduced to its best double couple.
    m_mean = np.einsum("n,nij->ij", posterior, m)
    centroid_sdr = np.array(moment_tensor.MomentTensor(m=m_mean).both_strike_dip_rake()[0], dtype=float)

    return {
        "obs": obs,
        "grid": grid,
        "log_likelihood": log_like,
        "posterior": posterior,
        "best_idx": best_idx,
        "best_sdr": best_sdr,
        "centroid_sdr": centroid_sdr,
        "sigma_i": sigma_i,
        "yi": yi,
    }


def preferred_sdr(cfg, result):
    choice = str(cfg["inversion"].get("preferred_solution", "centroid")).lower()
    if choice not in ("centroid", "best"):
        raise ValueError("inversion.preferred_solution must be 'centroid' or 'best', got {!r}".format(choice))
    return choice, result["{}_sdr".format(choice)]


def predicted_polarities(sdr, obs):
    m = sdr_to_moment_tensor(*sdr)
    amps = p_amplitudes(m[None], ray_vectors(obs["azimuth"], obs["takeoff_angle"]))[0]
    return np.where(amps >= 0.0, "U", "D"), amps


# ----------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------
def save_result(cfg, event_row, result, npz_path):
    obs = result["obs"]
    grid = result["grid"]
    label, pref = preferred_sdr(cfg, result)
    metadata = {
        "event_id": event_row["event_id"],
        "origin_time": event_row["event_datetime"],
        "event_lat": float(event_row["event_lat"]),
        "event_lon": float(event_row["event_lon"]),
        "event_depth_km": float(event_row["event_depth"]),
        "magnitude": float(event_row["event_mag"]),
        "velocity_model": cfg["velocity_model"]["model"],
        "nobs": int(len(obs)),
        "ngrid": int(len(grid)),
        "inversion": dict(cfg["inversion"]),
        "best_idx": result["best_idx"],
        "best_sdr": result["best_sdr"].tolist(),
        "centroid_sdr": result["centroid_sdr"].tolist(),
        "preferred_solution": label,
        "preferred_sdr": pref.tolist(),
    }
    np.savez_compressed(
        npz_path,
        strike=grid[:, 0],
        dip=grid[:, 1],
        rake=grid[:, 2],
        log_likelihood=result["log_likelihood"],
        posterior=result["posterior"],
        best_sdr=result["best_sdr"],
        centroid_sdr=result["centroid_sdr"],
        obs_station=obs["sta"].to_numpy(dtype=str),
        obs_network=obs["net"].to_numpy(dtype=str),
        obs_channel=obs["chan"].to_numpy(dtype=str),
        obs_polarity=obs["polarity"].to_numpy(dtype=str),
        obs_yi=result["yi"],
        obs_confidence=obs["confidence"].to_numpy(dtype=float),
        obs_sigma_i=result["sigma_i"],
        station_lat=obs["station_lat"].to_numpy(dtype=float),
        station_lon=obs["station_lon"].to_numpy(dtype=float),
        azimuth_deg=obs["azimuth"].to_numpy(dtype=float),
        takeoff_deg=obs["takeoff_angle"].to_numpy(dtype=float),
        metadata_json=json.dumps(metadata),
    )


def summary_text(cfg, event_row, result):
    obs = result["obs"]
    label, pref = preferred_sdr(cfg, result)
    plane1, plane2 = moment_tensor.MomentTensor(
        strike=pref[0], dip=pref[1], rake=pref[2]
    ).both_strike_dip_rake()

    pred, amps = predicted_polarities(pref, obs)
    misfit = pred != obs["polarity"].to_numpy()

    lines = [
        "Focal-mechanism polarity inversion",
        "==================================",
        "Event:           {}".format(event_row["event_id"]),
        "Origin time:     {}".format(event_row["event_datetime"]),
        "Location:        {:.4f}, {:.4f}, depth {:.1f} km".format(
            event_row["event_lat"], event_row["event_lon"], event_row["event_depth"]),
        "Magnitude:       {}".format(event_row["event_mag"]),
        "Velocity model:  {}".format(cfg["velocity_model"]["model"]),
        "Polarities used: {} (U: {}, D: {})".format(
            len(obs), int((obs["polarity"] == "U").sum()), int((obs["polarity"] == "D").sum())),
        "Grid points:     {}".format(len(result["grid"])),
        "",
        "Maximum-posterior grid solution (strike/dip/rake): {:.0f} / {:.0f} / {:.0f}".format(*result["best_sdr"]),
        "Posterior-mean (centroid) solution:                {:.0f} / {:.0f} / {:.0f}".format(*result["centroid_sdr"]),
        "",
        "Preferred solution ({}):".format(label),
        "    nodal plane 1: strike {:5.0f}  dip {:4.0f}  rake {:6.0f}".format(*plane1),
        "    nodal plane 2: strike {:5.0f}  dip {:4.0f}  rake {:6.0f}".format(*plane2),
        "",
        "Polarities inconsistent with the preferred solution: {} of {}".format(int(misfit.sum()), len(obs)),
        "",
        "{:<5s} {:<6s} {:>8s} {:>8s} {:>8s} {:>4s} {:>5s} {:>6s}  {}".format(
            "net", "sta", "dist_km", "azimuth", "takeoff", "obs", "pred", "conf", ""),
    ]
    for i, r in obs.iterrows():
        lines.append("{:<5s} {:<6s} {:8.1f} {:8.1f} {:8.1f} {:>4s} {:>5s} {:6.2f}  {}".format(
            r["net"], r["sta"], r["distance_km"], r["azimuth"], r["takeoff_angle"],
            r["polarity"], pred[i], r["confidence"], "<-- misfit" if misfit[i] else ""))
    return "\n".join(lines) + "\n"
