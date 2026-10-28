"""
Interactive polarity picking with Pyrocko Snuffler, and reading the
resulting marker files.

Snuffler marker conventions used in this tutorial:
    F1            turn the selected marker into a "P" phase marker
    Up arrow      polarity UP   (+1, compression)
    Down arrow    polarity DOWN (-1, dilatation)
    Shift+Up/Down polarity 0    (onset seen, polarity unclear)
    0..5          marker "kind" = pick confidence class (see config)
"""

import shutil

import pandas as pd
from pyrocko import pile as pile_mod
from pyrocko import util
from pyrocko.gui.snuffler import marker as pmarker

from .config import MissingInputError, require
from .waveforms import to_pyrocko_stations
from .velocity import first_arrival, load_velocity_model, phase_defs


PREDICTED_SUFFIX = "_pred"
PREDICTED_KIND = 6

POLARITY_SYMBOL = {1: "U", -1: "D", 0: "?"}


def is_predicted(m):
    return isinstance(m, pmarker.PhaseMarker) and (m.get_phasename() or "").endswith(PREDICTED_SUFFIX)


def predicted_markers(cfg, event, stations):
    """
    Phase markers at the predicted arrival times of the configured phases
    (e.g. P and S) for every station.
    """
    groups = cfg["picking"].get("predicted_phases") or {}
    if not groups:
        return []

    mod = load_velocity_model(cfg)
    depth_km = event.depth / 1000.0
    defs = {label: phase_defs(spec) for label, spec in groups.items()}

    markers = []
    for r in stations.itertuples():
        nslc = (r.network, r.station, r.location, "*")
        for label, phases in defs.items():
            arrival = first_arrival(mod, r.distance_km, depth_km, phases)
            if arrival is None:
                continue
            t = event.time + arrival.t
            markers.append(
                pmarker.PhaseMarker(
                    [nslc],
                    t,
                    t,
                    kind=PREDICTED_KIND,
                    event=event,
                    phasename=label + PREDICTED_SUFFIX,
                )
            )
    return markers


def load_user_markers(path, event):
    """
    Load previously saved picks for this event (predicted and event
    markers are dropped; the picks are re-attached to ``event``).
    """
    if not path.is_file():
        return []
    markers = pmarker.load_markers(str(path))
    picks = [m for m in markers if not isinstance(m, pmarker.EventMarker) and not is_predicted(m)]
    for m in picks:
        if isinstance(m, pmarker.PhaseMarker):
            m.set_event(event)
            m.set_event_hash(None)
    return picks


def processed_files(cfg, event_id):
    require(cfg.processed_dir(event_id), "Processed waveforms", "run 02_download_waveforms.py first")
    components = str(cfg["picking"].get("components", "Z")).upper()
    files = []
    for f in sorted(cfg.processed_dir(event_id).glob("*.mseed")):
        channel = f.stem.split(".")[-1]
        if channel[-1:].upper() in components:
            files.append(str(f))
    return files


def _make_launch_hook(cfg, event, predicted):
    pick_cfg = cfg["picking"]

    def hook(win):
        view = win.get_view()

        if pick_cfg.get("sort_by_distance", True):
            try:
                for item, _ in view.menuitems_ssorting:
                    if item.text().replace("&", "") == "Sort by Distance":
                        item.setChecked(True)
                view.s_sortingmode_change()
            except Exception as exc:
                print("Note: could not set distance sorting automatically ({}).".format(exc))
                print("      Use the menu: Sort by Distance (keys: S then D).")

        # Start with a window from just before the origin to after the
        # latest predicted arrival.
        try:
            tmax = max([m.tmax for m in predicted] + [event.time + 60.0])
            view.set_time_range(event.time - 10.0, tmax + 20.0)
        except Exception:
            pass

    return hook


def run_snuffler(cfg, event, stations):
    """
    Open Snuffler for one event. Returns the list of user markers present
    when the window was closed.
    """
    from pyrocko.gui.snuffler.snuffler import snuffle

    event_id = event.name
    files = processed_files(cfg, event_id)
    if not files:
        raise MissingInputError(
            "No processed waveforms for components {!r} in {}".format(
                cfg["picking"].get("components", "Z"), cfg.processed_dir(event_id)
            )
        )

    markers_path = cfg.markers_file(event_id)
    user_markers = load_user_markers(markers_path, event)
    predicted = predicted_markers(cfg, event, stations)

    pyrocko_stations = to_pyrocko_stations(stations)

    p = pile_mod.make_pile(files, show_progress=False)

    print("Loaded {} traces, {} stations, {} existing pick(s), {} predicted marker(s).".format(
        len(files), len(pyrocko_stations), len(user_markers), len(predicted)))

    _, markers = snuffle(
        p,
        stations=pyrocko_stations,
        events=[event],
        markers=user_markers + predicted,
        ntracks=int(cfg["picking"].get("ntracks", 12)),
        want_markers=True,
        launch_hook=_make_launch_hook(cfg, event, predicted),
    )

    return [m for m in markers if not is_predicted(m)]


def save_markers(cfg, event_id, markers):
    """
    Save user markers (event marker + picks). An existing file is kept as
    <name>.bak before being overwritten.
    """
    path = cfg.markers_file(event_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        shutil.copy2(path, path.with_suffix(".txt.bak"))
    pmarker.save_markers(markers, str(path))
    return path


def polarity_table(cfg, markers):
    """
    Phase markers of the configured polarity phases as a DataFrame with
    one row per pick.
    """
    pick_cfg = cfg["picking"]
    phases = set(pick_cfg.get("polarity_phases", ["P"]))
    conf_by_kind = {int(k): float(v) for k, v in (pick_cfg.get("confidence_by_kind") or {}).items()}

    rows = []
    for m in markers:
        if not isinstance(m, pmarker.PhaseMarker) or m.get_phasename() not in phases:
            continue
        for net, sta, loc, cha in m.get_nslc_ids():
            pol = m.get_polarity()
            rows.append(
                {
                    "network": net,
                    "station": sta,
                    "location": loc,
                    "channel": cha,
                    "phase": m.get_phasename(),
                    "pick_time": util.time_to_str(m.tmin),
                    "polarity": POLARITY_SYMBOL.get(pol, ""),
                    "kind": m.kind,
                    "confidence": conf_by_kind.get(m.kind, float("nan")),
                }
            )

    return pd.DataFrame(
        rows,
        columns=["network", "station", "location", "channel", "phase", "pick_time", "polarity", "kind", "confidence"],
    )


def read_polarity_picks(cfg, event_id):
    path = require(
        cfg.markers_file(event_id),
        "Polarity picks (Snuffler marker file)",
        "run 03_pick_polarities.py and pick polarities first",
    )
    return polarity_table(cfg, pmarker.load_markers(str(path)))
