#!/usr/bin/env python3
"""
Step 2: download and pre-process waveforms for each event.

For every event in <data_dir>/catalog/events.csv:
  1. download raw MiniSEED + StationXML for stations within the configured
     distance range from the configured FDSN data centres,
  2. remove the instrument response, rotate to Z/N/E and make sure the
     vertical component is positive UP,
  3. write the processed data, a station table and a station coverage map.

Output per event (<data_dir>/events/<event_id>/):
    raw/  stationxml/  processed/  stations.csv  stations.pf
    event.pf  station_map.png

Examples:
    python 02_download_waveforms.py --config config.yaml
    python 02_download_waveforms.py --config config.yaml --event 20020817.160627
    python 02_download_waveforms.py --config config.yaml --skip-download
"""

import traceback

from fmtools.catalog import read_event_table
from fmtools.config import MissingInputError, add_event_selection_arguments, base_argument_parser, load_config, run
from fmtools.waveforms import (
    download_event_waveforms,
    azimuthal_gap,
    plot_station_map,
    preprocess_event_waveforms,
)


def main():
    parser = base_argument_parser(__doc__)
    add_event_selection_arguments(parser)
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Only re-run pre-processing on already downloaded raw data.",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)

    events = read_event_table(cfg, args.event)
    print("Processing {} event(s).".format(len(events)))

    failed = []
    for i, row in events.iterrows():
        event_id = row["event_id"]
        print("\n" + "=" * 70)
        print("[{}/{}] Event {}  M{} {}  ({:.3f}, {:.3f}, {:.1f} km)".format(
            i + 1, len(events), event_id, row["event_mag"], row["event_mag_type"],
            row["event_lat"], row["event_lon"], row["event_depth"]))
        print("=" * 70)

        try:
            if not args.skip_download:
                print("Downloading raw data -> {}".format(cfg.raw_dir(event_id)))
                download_event_waveforms(cfg, row)

            print("Pre-processing -> {}".format(cfg.processed_dir(event_id)))
            stations = preprocess_event_waveforms(cfg, row)

            if stations.empty:
                print("No usable stations for this event.")
                failed.append(event_id)
                continue

            map_file = cfg.event_dir(event_id) / "station_map.png"
            plot_station_map(stations, row, map_file)

            print("\n{} station(s), {:.0f}-{:.0f} km, largest azimuthal gap {:.0f} deg.".format(
                len(stations), stations["distance_km"].min(), stations["distance_km"].max(),
                azimuthal_gap(stations["azimuth"])))
            print("Station table: {}".format(cfg.stations_csv(event_id)))
            print("Station map: {}".format(map_file))

        except MissingInputError as exc:
            print("SKIP: {}".format(exc))
            failed.append(event_id)
        except Exception:
            traceback.print_exc()
            failed.append(event_id)

    print("\nDone. {} event(s) OK, {} failed{}".format(
        len(events) - len(failed), len(failed), ": " + ", ".join(failed) if failed else "."))


if __name__ == "__main__":
    run(main)
