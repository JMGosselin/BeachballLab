"""
Event search (FDSN) and the tutorial's event table.

The event table (catalog/events.csv) is the hand-off between steps: every
later script reads it to know which events exist and where they are.
"""

import pandas as pd
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from obspy.geodetics import kilometer2degrees
from pyrocko import model

from .config import require


EVENT_COLUMNS = [
    "event_id",
    "event_datetime",
    "event_lat",
    "event_lon",
    "event_depth",
    "event_mag",
    "event_mag_type",
    "catalog_id",
]


def _optional(value):
    return None if value is None else float(value)


def search_events(cfg):
    """
    Query an FDSN event service using the event_search section of the config.
    Returns an ObsPy Catalog sorted by origin time.
    """
    es = cfg["event_search"]

    kwargs = dict(
        starttime=UTCDateTime(es["starttime"]),
        endtime=UTCDateTime(es["endtime"]),
        minmagnitude=_optional(es.get("min_magnitude")),
        maxmagnitude=_optional(es.get("max_magnitude")),
        mindepth=_optional(es.get("min_depth_km")),
        maxdepth=_optional(es.get("max_depth_km")),
        orderby="time-asc",
    )

    region_type = es.get("region_type", "circle").lower()
    if region_type == "circle":
        kwargs.update(
            latitude=float(es["center_lat"]),
            longitude=float(es["center_lon"]),
            maxradius=kilometer2degrees(float(es["max_radius_km"])),
        )
    elif region_type == "box":
        kwargs.update(
            minlatitude=float(es["min_latitude"]),
            maxlatitude=float(es["max_latitude"]),
            minlongitude=float(es["min_longitude"]),
            maxlongitude=float(es["max_longitude"]),
        )
    else:
        raise ValueError("event_search.region_type must be 'circle' or 'box', got {!r}".format(region_type))

    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    print("Querying FDSN event service: {}".format(es["client"]))
    for k, v in kwargs.items():
        print("    {:<14s} {}".format(k, v))

    client = Client(es["client"])
    catalog = client.get_events(**kwargs)
    catalog.events.sort(key=lambda ev: (ev.preferred_origin() or ev.origins[0]).time)

    max_events = es.get("max_events")
    if max_events is not None:
        catalog.events = catalog.events[: int(max_events)]

    return catalog


def make_event_id(origin_time):
    """
    Event ids are based on origin time, e.g. 20020817.160627.
    """
    return UTCDateTime(origin_time).strftime("%Y%m%d.%H%M%S")


def catalog_to_table(catalog):
    rows = []
    used_ids = set()
    for ev in catalog:
        origin = ev.preferred_origin() or ev.origins[0]
        mag = ev.preferred_magnitude() or (ev.magnitudes[0] if ev.magnitudes else None)

        event_id = make_event_id(origin.time)
        # Two events in the same second would otherwise collide.
        suffix = 1
        base_id = event_id
        while event_id in used_ids:
            event_id = "{}_{}".format(base_id, suffix)
            suffix += 1
        used_ids.add(event_id)

        rows.append(
            {
                "event_id": event_id,
                "event_datetime": str(origin.time),
                "event_lat": origin.latitude,
                "event_lon": origin.longitude,
                "event_depth": origin.depth / 1000.0 if origin.depth is not None else float("nan"),
                "event_mag": mag.mag if mag is not None else float("nan"),
                "event_mag_type": mag.magnitude_type if mag is not None else "",
                "catalog_id": str(ev.resource_id),
            }
        )

    return pd.DataFrame(rows, columns=EVENT_COLUMNS)


def read_event_table(cfg, event_ids=None):
    """
    Read catalog/events.csv, optionally restricted to a list of event ids.
    """
    path = require(cfg.catalog_csv, "Event table", "run 01_search_events.py first")

    df = pd.read_csv(path, dtype={"event_id": str})

    if event_ids:
        unknown = sorted(set(event_ids) - set(df["event_id"]))
        if unknown:
            raise ValueError(
                "Event id(s) not in {}: {}".format(path, ", ".join(unknown))
            )
        df = df[df["event_id"].isin(event_ids)]

    return df.reset_index(drop=True)


def row_to_pyrocko_event(row):
    mag = row["event_mag"]
    return model.Event(
        name=row["event_id"],
        lat=float(row["event_lat"]),
        lon=float(row["event_lon"]),
        depth=float(row["event_depth"]) * 1000.0,
        time=UTCDateTime(row["event_datetime"]).timestamp,
        magnitude=None if pd.isna(mag) else float(mag),
        magnitude_type=row["event_mag_type"] if isinstance(row["event_mag_type"], str) else None,
    )
