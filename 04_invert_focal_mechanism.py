#!/usr/bin/env python3
"""
Step 4: invert the picked polarities for a focal mechanism.

For every event that has a Snuffler marker file (step 3):
  1. combine the P polarity picks with the station table and compute
     take-off angles with the velocity model,
  2. run a probabilistic grid search over strike, dip and rake,
  3. write the results and a fuzzy beachball plot with the polarities.

Events without a marker file, or with fewer than inversion.min_polarities
usable (U/D) polarities, are skipped.

Output per event (<data_dir>/events/<event_id>/inversion/):
    <event_id>_polarities.csv        picks with azimuth and take-off angle
    <event_id>_dc_grid_ppd.npz       full grid and posterior
    <event_id>_inversion_summary.txt solutions and per-station fit
    <event_id>_fuzzy_beachball.png   fuzzy beachball + polarities

Examples:
    python 04_invert_focal_mechanism.py --config config.yaml
    python 04_invert_focal_mechanism.py --config config.yaml --event 20020817.160627
    python 04_invert_focal_mechanism.py --config config.yaml --no-plot
"""

import time
import traceback

from fmtools.catalog import read_event_table
from fmtools.config import MissingInputError, add_event_selection_arguments, base_argument_parser, load_config, run
from fmtools.inversion import (
    build_polarity_table,
    invert_polarities,
    preferred_sdr,
    save_result,
    summary_text,
)
from fmtools.picking import read_polarity_picks
from fmtools.waveforms import read_station_table


def main():
    parser = base_argument_parser(__doc__)
    add_event_selection_arguments(parser)
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Do not make the fuzzy beachball plot.",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)

    events = read_event_table(cfg, args.event)
    min_pol = int(cfg["inversion"].get("min_polarities", 8))

    done, skipped, failed = [], [], []
    for _, row in events.iterrows():
        event_id = row["event_id"]
        print("\n" + "=" * 70)
        print("Event {}".format(event_id))
        print("=" * 70)

        try:
            picks = read_polarity_picks(cfg, event_id)
            stations = read_station_table(cfg, event_id)
        except MissingInputError as exc:
            print("SKIP: {}".format(exc))
            skipped.append(event_id)
            continue

        try:
            cfg.inversion_dir(event_id).mkdir(parents=True, exist_ok=True)

            table = build_polarity_table(cfg, row, picks, stations)
            table_file = cfg.inversion_file(event_id, "polarities.csv")
            table.to_csv(table_file, index=False)

            n_used = int(table["accepted"].sum())
            print("Picks: {}, usable U/D polarities: {}  -> {}".format(len(table), n_used, table_file))
            if n_used < min_pol:
                print("SKIP: only {} usable polarities (inversion.min_polarities = {}).".format(n_used, min_pol))
                skipped.append(event_id)
                continue

            t0 = time.time()
            result = invert_polarities(cfg, table)
            print("Grid search over {} mechanisms took {:.1f} s.".format(len(result["grid"]), time.time() - t0))

            save_result(cfg, row, result, cfg.inversion_file(event_id, "dc_grid_ppd.npz"))

            summary = summary_text(cfg, row, result)
            cfg.inversion_file(event_id, "inversion_summary.txt").write_text(summary)
            print()
            print(summary)

            if not args.no_plot:
                from fmtools.beachball_plot import plot_fuzzy_beachball

                label, sdr = preferred_sdr(cfg, result)
                plot_file = cfg.inversion_file(event_id, "fuzzy_beachball.png")
                plot_fuzzy_beachball(cfg, row, result, sdr, label, plot_file)
                print("Fuzzy beachball: {}".format(plot_file))

            done.append(event_id)

        except Exception:
            traceback.print_exc()
            failed.append(event_id)

    print("\nDone. Inverted: {}, skipped: {}, failed: {}".format(len(done), len(skipped), len(failed)))
    for label, ids in (("Skipped", skipped), ("Failed", failed)):
        if ids:
            print("{}: {}".format(label, ", ".join(ids)))


if __name__ == "__main__":
    run(main)
