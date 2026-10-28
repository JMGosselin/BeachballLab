"""
Waveform download from FDSN data centres and pre-processing.

Raw data and station metadata are stored per event; pre-processed data are
written to a separate folder so the raw data are never modified.
"""

import fnmatch
import shutil
import time
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
from obspy import Stream, UTCDateTime, read, read_inventory
from obspy.clients.fdsn import Client
from obspy.clients.fdsn.header import (
    FDSNNoDataException,
    FDSNServiceUnavailableException,
    FDSNTimeoutException,
    FDSNTooManyRequestsException,
)
from obspy.geodetics import gps2dist_azimuth, kilometer2degrees
from pyrocko import model

from .config import require


# ----------------------------------------------------------------------
# Download
# ----------------------------------------------------------------------
def _fdsn_channel_query(channel_priorities):
    """
    ["HH[ZNE12]", "BH[ZNE12]"] -> "HH?,BH?" (FDSN only understands ? and *).
    """
    return ",".join(sorted({p[:2] + "?" for p in channel_priorities}))


def _select_channels(station, channel_priorities, location_priorities):
    """
    Pick the preferred channel group of one station: the first channel
    pattern (and location code) in the priority lists with data.
    Returns (location, [channel codes]) or None.
    """
    for cha_pattern in channel_priorities:
        for loc in location_priorities:
            loc = "" if loc == "--" else loc
            codes = sorted(
                {
                    ch.code
                    for ch in station
                    if ch.location_code == loc and fnmatch.fnmatch(ch.code, cha_pattern)
                }
            )
            if codes:
                return loc, codes
    return None


def _quality_check(st, t1, t2, reject_gaps, minimum_length):
    """
    Drop channels with gaps or with too little data. Returns kept Stream
    and a list of messages about dropped channels.
    """
    kept = Stream()
    dropped = []
    for seed_id in sorted({tr.id for tr in st}):
        sub = st.select(id=seed_id)
        sub.sort(["starttime"])
        if reject_gaps and len(sub.get_gaps()) > 0:
            dropped.append("{} (gaps)".format(seed_id))
            continue
        sub.merge(method=1, fill_value="interpolate")
        sub.trim(t1, t2)
        covered = sum(tr.stats.endtime - tr.stats.starttime for tr in sub)
        if covered < minimum_length * (t2 - t1):
            dropped.append("{} (only {:.0f} s)".format(seed_id, covered))
            continue
        kept += sub
    return kept, dropped


def _short_error(exc):
    """
    First line of an exception message (servers often return whole HTML pages).
    """
    lines = str(exc).strip().splitlines()
    return lines[0] if lines else exc.__class__.__name__


class ThrottledClient:
    """
    FDSN client that waits at least ``interval`` seconds between requests
    and retries when the server is busy. Some data centres (e.g. NRCan:
    max. one request per 5 s) refuse requests that come too quickly.
    """

    RETRY_ON = (
        FDSNServiceUnavailableException,
        FDSNTooManyRequestsException,
        FDSNTimeoutException,
    )

    def __init__(self, provider, timeout=180.0, interval=0.0, retries=3):
        # Service discovery is skipped: it is slow or fails on some
        # servers, and FDSN servers use the standard paths.
        self.client = Client(provider, timeout=timeout, _discover_services=False)
        self.interval = interval
        self.retries = retries
        self._last = 0.0

    def _call(self, method, *args, **kwargs):
        for attempt in range(self.retries + 1):
            wait = self.interval - (time.time() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.time()
            try:
                return getattr(self.client, method)(*args, **kwargs)
            except self.RETRY_ON as exc:
                if attempt == self.retries:
                    raise
                pause = max(self.interval, 5.0) * (attempt + 2)
                print("        server busy ({}); retrying in {:.0f} s".format(_short_error(exc), pause))
                time.sleep(pause)

    def get_stations(self, *args, **kwargs):
        return self._call("get_stations", *args, **kwargs)

    def get_waveforms(self, *args, **kwargs):
        return self._call("get_waveforms", *args, **kwargs)


def _fetch_waveforms(client, network, selections, t1, t2):
    """
    Request waveforms for several stations of one network in a single
    request (much faster than one request per station), falling back to
    one request per station if that fails. ``selections`` maps station
    code -> (location, [channels]). Only the selected channels are returned.
    """
    def wanted(tr):
        sel = selections.get(tr.stats.station)
        return sel is not None and tr.stats.location == sel[0] and tr.stats.channel in sel[1]

    channels = sorted({c for _, chans in selections.values() for c in chans})
    locations = sorted({loc or "--" for loc, _ in selections.values()})
    try:
        st = client.get_waveforms(
            network, ",".join(sorted(selections)), ",".join(locations), ",".join(channels), t1, t2
        )
        return Stream([tr for tr in st if wanted(tr)])
    except FDSNNoDataException:
        return Stream()
    except Exception as exc:
        if len(selections) == 1:
            raise
        print("      network request for {} failed ({}); trying station by station".format(
            network, _short_error(exc)))

    st = Stream()
    for code, (loc, chans) in sorted(selections.items()):
        try:
            st += client.get_waveforms(network, code, loc or "--", ",".join(chans), t1, t2)
        except FDSNNoDataException:
            pass
        except Exception as exc:
            print("        {}.{} waveform request failed: {}".format(network, code, _short_error(exc)))
    return st


def download_event_waveforms(cfg, event_row):
    """
    Download raw MiniSEED and StationXML for one event.

    The providers in the config are queried in order; a station that was
    already obtained from an earlier provider (or in an earlier run) is
    skipped. Requests are grouped by network to keep their number small.

    Writes raw/<NET.STA.LOC.CHA>.mseed and stationxml/<NET.STA>.xml.
    """
    event_id = event_row["event_id"]
    sta_cfg = cfg["stations"]
    wf_cfg = cfg["waveforms"]

    raw_dir = cfg.raw_dir(event_id)
    sxml_dir = cfg.stationxml_dir(event_id)

    if wf_cfg.get("overwrite", False):
        for d in (raw_dir, sxml_dir):
            if d.exists():
                shutil.rmtree(d)

    raw_dir.mkdir(parents=True, exist_ok=True)
    sxml_dir.mkdir(parents=True, exist_ok=True)

    origin_time = UTCDateTime(event_row["event_datetime"])
    t1 = origin_time - float(wf_cfg["time_before_origin"])
    t2 = origin_time + float(wf_cfg["time_after_origin"])

    channel_priorities = sta_cfg["channel_priorities"]
    location_priorities = sta_cfg.get("location_priorities", ["", "00", "10"])
    reject_gaps = bool(wf_cfg.get("reject_channels_with_gaps", True))
    minimum_length = float(wf_cfg.get("minimum_length", 0.9))

    # Stations already on disk (earlier run) are not downloaded again.
    done = {tuple(f.stem.split(".")[:2]) for f in sxml_dir.glob("*.xml")}
    if done:
        print("    {} station(s) already downloaded.".format(len(done)))

    for provider in sta_cfg["providers"]:
        print("    Provider {}".format(provider))
        try:
            client = ThrottledClient(
                provider,
                timeout=float(sta_cfg.get("timeout_s", 180)),
                interval=float(sta_cfg.get("request_interval_s", 0.0)),
            )
            inventory = client.get_stations(
                latitude=float(event_row["event_lat"]),
                longitude=float(event_row["event_lon"]),
                minradius=kilometer2degrees(float(sta_cfg.get("min_radius_km", 0.0))),
                maxradius=kilometer2degrees(float(sta_cfg["max_radius_km"])),
                network=sta_cfg.get("network", "*"),
                station=sta_cfg.get("station", "*"),
                channel=_fdsn_channel_query(channel_priorities),
                starttime=t1,
                endtime=t2,
                level="channel",
            )
        except FDSNNoDataException:
            print("      no stations found")
            continue
        except Exception as exc:
            print("      station query failed: {}".format(_short_error(exc)))
            continue

        for net in inventory:
            selections = {}
            for sta in net:
                if (net.code, sta.code) in done:
                    continue
                selection = _select_channels(sta, channel_priorities, location_priorities)
                if selection is not None:
                    selections[sta.code] = selection
            if not selections:
                continue

            print("      {}: requesting {} station(s) ...".format(net.code, len(selections)))
            try:
                st_net = _fetch_waveforms(client, net.code, selections, t1, t2)
            except Exception as exc:
                print("      {}: waveform request failed: {}".format(net.code, _short_error(exc)))
                continue

            # Quality control per station
            kept = {}
            for code, (loc, chans) in sorted(selections.items()):
                label = "{}.{}.{}.{}".format(net.code, code, loc, ",".join(chans))
                st = st_net.select(station=code)
                if len(st) == 0:
                    print("        {:<28s} no data".format(label))
                    continue
                st, dropped = _quality_check(st, t1, t2, reject_gaps, minimum_length)
                if len(st) == 0:
                    print("        {:<28s} rejected: {}".format(label, ", ".join(dropped)))
                    continue
                kept[code] = (label, st, dropped)

            if not kept:
                continue

            # Instrument responses for all kept stations of this network
            try:
                inv_resp = client.get_stations(
                    network=net.code,
                    station=",".join(sorted(kept)),
                    location=",".join(sorted({tr.stats.location or "--" for _, st, _ in kept.values() for tr in st})),
                    channel=",".join(sorted({tr.stats.channel for _, st, _ in kept.values() for tr in st})),
                    starttime=t1,
                    endtime=t2,
                    level="response",
                )
            except Exception as exc:
                print("      {}: response request failed: {}".format(net.code, _short_error(exc)))
                continue

            for code, (label, st, dropped) in sorted(kept.items()):
                inv_sta = inv_resp.select(station=code)
                if len(inv_sta) == 0:
                    print("        {:<28s} no response metadata".format(label))
                    continue
                for tr in st:
                    tr.write(str(raw_dir / "{}.mseed".format(tr.id)), format="MSEED")
                inv_sta.write(str(sxml_dir / "{}.{}.xml".format(net.code, code)), format="STATIONXML")
                done.add((net.code, code))

                msg = "        {:<28s} OK".format(label)
                if dropped:
                    msg += "  (dropped {})".format(", ".join(dropped))
                print(msg)


# ----------------------------------------------------------------------
# Pre-processing
# ----------------------------------------------------------------------
def _pre_filt_for_trace(tr, pre_filt):
    if pre_filt is None:
        return None
    if isinstance(pre_filt, str) and pre_filt.lower() == "auto":
        fs = tr.stats.sampling_rate
        return [0.01, 0.02, 0.4 * fs, 0.45 * fs]
    return [float(f) for f in pre_filt]


def _read_raw(cfg, event_id):
    hint = "run 02_download_waveforms.py without --skip-download first"
    require(cfg.raw_dir(event_id), "Raw waveforms", hint)
    require(cfg.stationxml_dir(event_id), "StationXML metadata", hint)
    raw_files = sorted(cfg.raw_dir(event_id).glob("*.mseed"))
    sxml_files = sorted(cfg.stationxml_dir(event_id).glob("*.xml"))

    st = Stream()
    for f in raw_files:
        try:
            st += read(str(f))
        except Exception as exc:
            print("    WARNING: could not read {}: {}".format(f.name, exc))

    inv = None
    for f in sxml_files:
        try:
            this_inv = read_inventory(str(f))
        except Exception as exc:
            print("    WARNING: could not read {}: {}".format(f.name, exc))
            continue
        inv = this_inv if inv is None else inv + this_inv

    return st, inv


def _channel_dip(inv, seed_id, time):
    try:
        return inv.get_orientation(seed_id, time)["dip"]
    except Exception:
        return None


def _orient_group(group, inv, rotate_to_zne):
    """
    Bring one station/instrument group to a Z-up (Z, N, E) system.

    Stations with 1/2(/3) components are rotated using their metadata.
    For a Z channel that is not rotated, the dip is checked: a vertical
    sensor installed/registered with dip = +90 (pointing down) is flipped
    so that positive values always mean UP motion.
    """
    codes = {tr.stats.channel[-1] for tr in group}
    t_ref = group[0].stats.starttime
    notes = []

    if rotate_to_zne and ({"1", "2"} <= codes or {"1", "2", "3"} <= codes):
        try:
            group.rotate("->ZNE", inventory=inv, components=("Z12", "123"))
            notes.append("rotated to ZNE")
            return group, notes
        except Exception as exc:
            notes.append("rotation failed ({})".format(exc))

    for tr in group:
        if tr.stats.channel[-1] != "Z":
            continue
        dip = _channel_dip(inv, tr.id, t_ref)
        if dip is not None and dip > 0:
            tr.data = -tr.data
            notes.append("{} has dip {:+.0f}; flipped to Z-up".format(tr.id, dip))

    return group, notes


def preprocess_event_waveforms(cfg, event_row):
    """
    Pre-process raw data for one event and write:
        processed/<NET.STA.LOC.CHA>.mseed
        stations.csv, stations.pf
    Returns the station table as a DataFrame.
    """
    event_id = event_row["event_id"]
    pp = cfg["preprocessing"]

    st, inv = _read_raw(cfg, event_id)
    if inv is None:
        raise RuntimeError("No readable StationXML for event {}".format(event_id))

    out_dir = cfg.processed_dir(event_id)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    st.merge(method=1, fill_value="interpolate")

    # Group by network/station/location/band+instrument (e.g. CN.PNT..BH)
    groups = defaultdict(Stream)
    for tr in st:
        s = tr.stats
        groups[(s.network, s.station, s.location, s.channel[:2])].append(tr)

    processed = Stream()
    for key in sorted(groups):
        group = groups[key]
        label = "{}.{}.{}.{}?".format(*key)
        try:
            group.detrend("demean")
            group.detrend("linear")
            group.taper(max_percentage=0.05, type="cosine")

            if pp.get("remove_response", True):
                for tr in group:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        tr.remove_response(
                            inventory=inv,
                            output=pp.get("output", "VEL"),
                            pre_filt=_pre_filt_for_trace(tr, pp.get("pre_filt", "auto")),
                            water_level=pp.get("water_level", 60.0),
                        )

            group, notes = _orient_group(group, inv, pp.get("rotate_to_zne", True))

            bandpass = pp.get("bandpass")
            if bandpass:
                group.filter(
                    "bandpass",
                    freqmin=float(bandpass[0]),
                    freqmax=float(bandpass[1]),
                    corners=4,
                    zerophase=False,
                )

            if pp.get("resample_hz"):
                group.resample(float(pp["resample_hz"]))

        except Exception as exc:
            print("    SKIP {:<16s} {}".format(label, exc))
            continue

        msg = "    OK   {:<16s} {}".format(label, ", ".join(tr.stats.channel for tr in group))
        if notes:
            msg += "  [{}]".format("; ".join(notes))
        print(msg)
        processed += group

    for tr in processed:
        tr.data = tr.data.astype(np.float32)
        tr.write(str(out_dir / "{}.mseed".format(tr.id)), format="MSEED", encoding="FLOAT32")

    stations = station_table(processed, inv, event_row)
    stations.to_csv(cfg.stations_csv(event_id), index=False, float_format="%.5f")
    write_pyrocko_stations(stations, cfg.stations_pf(event_id))

    model.dump_events(
        [_pyrocko_event(event_row)],
        filename=str(cfg.event_pf(event_id)),
    )

    return stations


def _pyrocko_event(event_row):
    # Local import avoids a circular import at module load time.
    from .catalog import row_to_pyrocko_event

    return row_to_pyrocko_event(event_row)


# ----------------------------------------------------------------------
# Station geometry
# ----------------------------------------------------------------------
def station_table(stream, inv, event_row):
    """
    One row per station (network, station, location) with coordinates and
    event-station geometry. Used by Snuffler now and by the take-off angle
    calculation later.
    """
    ev_lat = float(event_row["event_lat"])
    ev_lon = float(event_row["event_lon"])

    rows = {}
    for tr in stream:
        s = tr.stats
        key = (s.network, s.station, s.location)
        if key in rows:
            rows[key]["channels"].add(s.channel)
            continue
        try:
            coords = inv.get_coordinates(tr.id, s.starttime)
        except Exception:
            # Rotated traces have new channel codes; look up via the Z/any channel.
            sel = inv.select(network=s.network, station=s.station, location=s.location, time=s.starttime)
            try:
                ch = sel[0][0][0]
                coords = {"latitude": ch.latitude, "longitude": ch.longitude, "elevation": ch.elevation, "local_depth": ch.depth}
            except IndexError:
                print("    WARNING: no coordinates for {}".format(tr.id))
                continue

        dist_m, az, baz = gps2dist_azimuth(ev_lat, ev_lon, coords["latitude"], coords["longitude"])
        rows[key] = {
            "network": s.network,
            "station": s.station,
            "location": s.location,
            "station_lat": coords["latitude"],
            "station_lon": coords["longitude"],
            "station_elevation_m": coords["elevation"],
            "station_depth_m": coords.get("local_depth", 0.0) or 0.0,
            "distance_km": dist_m / 1000.0,
            "azimuth": az,
            "backazimuth": baz,
            "channels": {s.channel},
        }

    df = pd.DataFrame(rows.values())
    if df.empty:
        return df
    df["channels"] = df["channels"].apply(lambda c: ",".join(sorted(c)))
    return df.sort_values("distance_km").reset_index(drop=True)


def to_pyrocko_stations(stations):
    return [
        model.Station(
            network=r.network,
            station=r.station,
            location=r.location if isinstance(r.location, str) else "",
            lat=float(r.station_lat),
            lon=float(r.station_lon),
            elevation=float(r.station_elevation_m),
            depth=float(r.station_depth_m),
        )
        for r in stations.itertuples()
    ]


def write_pyrocko_stations(stations, filename):
    model.dump_stations(to_pyrocko_stations(stations), str(filename))


def read_station_table(cfg, event_id):
    path = require(cfg.stations_csv(event_id), "Station table", "run 02_download_waveforms.py first")
    return pd.read_csv(path, dtype={"network": str, "station": str, "location": str}, keep_default_na=False)


def azimuthal_gap(azimuths):
    """
    Largest gap (degrees) between neighbouring station azimuths.
    """
    az = np.sort(np.mod(np.asarray(azimuths, dtype=float), 360.0))
    if az.size == 0:
        return 360.0
    gaps = np.diff(np.concatenate([az, [az[0] + 360.0]]))
    return float(gaps.max())


def plot_station_map(stations, event_row, filename):
    """
    Simple map of station coverage around the epicentre. Good azimuthal
    coverage is essential for a well-constrained focal mechanism.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ev_lat = float(event_row["event_lat"])
    ev_lon = float(event_row["event_lon"])

    fig, (ax_map, ax_pol) = plt.subplots(
        1, 2, figsize=(12, 5.5), gridspec_kw={"width_ratios": [1.3, 1]}
    )
    ax_pol.remove()
    ax_pol = fig.add_subplot(1, 2, 2, projection="polar")

    ax_map.scatter(stations["station_lon"], stations["station_lat"], marker="^", s=60, c="tab:blue", edgecolor="k", zorder=3)
    for r in stations.itertuples():
        ax_map.annotate(
            "{}.{}".format(r.network, r.station),
            (r.station_lon, r.station_lat),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=7,
        )
    ax_map.scatter([ev_lon], [ev_lat], marker="*", s=300, c="tab:red", edgecolor="k", zorder=4)
    ax_map.set_aspect(1.0 / np.cos(np.deg2rad(ev_lat)))
    ax_map.set_xlabel("Longitude")
    ax_map.set_ylabel("Latitude")
    ax_map.set_title("Event {} and {} stations".format(event_row["event_id"], len(stations)))
    ax_map.grid(alpha=0.3)

    ax_pol.set_theta_zero_location("N")
    ax_pol.set_theta_direction(-1)
    ax_pol.scatter(np.deg2rad(stations["azimuth"]), stations["distance_km"], marker="^", c="tab:blue", edgecolor="k")
    ax_pol.set_title(
        "Azimuth and distance (km) from the epicentre\nlargest azimuthal gap: {:.0f} deg".format(
            azimuthal_gap(stations["azimuth"])),
        pad=25,
    )

    fig.tight_layout()
    fig.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
