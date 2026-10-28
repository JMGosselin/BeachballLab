#!/usr/bin/env python3
"""
Step 1: search for earthquakes.

Queries an FDSN event service using the `event_search` section of the
config file and writes the event list used by all later steps:

    <data_dir>/catalog/events.csv   (one row per event; edit to drop events)
    <data_dir>/catalog/events.xml   (QuakeML, full catalogue information)
    <data_dir>/catalog/events.pf    (Pyrocko event file)

Example:
    python 01_search_events.py --config config.yaml
"""

from fmtools.catalog import catalog_to_table, row_to_pyrocko_event, search_events
from fmtools.config import base_argument_parser, load_config, run
from pyrocko import model


def main():
    parser = base_argument_parser(__doc__)
    args = parser.parse_args()
    cfg = load_config(args.config)

    catalog = search_events(cfg)
    table = catalog_to_table(catalog)

    if table.empty:
        print("\nNo events found. Widen the search in the event_search section of the config.")
        return

    cfg.catalog_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(cfg.catalog_csv, index=False)
    catalog.write(str(cfg.catalog_dir / "events.xml"), format="QUAKEML")
    model.dump_events(
        [row_to_pyrocko_event(r) for _, r in table.iterrows()],
        filename=str(cfg.catalog_dir / "events.pf"),
    )

    print("\nFound {} event(s):\n".format(len(table)))
    print(table[["event_id", "event_datetime", "event_lat", "event_lon", "event_depth", "event_mag", "event_mag_type"]].to_string(index=False))
    print("\nWrote {}".format(cfg.catalog_csv))
    print("Tip: delete rows from events.csv to skip events in the next steps.")


if __name__ == "__main__":
    run(main)
