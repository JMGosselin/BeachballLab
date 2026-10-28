#!/usr/bin/env python3
"""
Step 5: post-process the focal-mechanism inversion.

For every event with inversion results (step 4):
  * P, T and N axis orientations and faulting style of the preferred solution,
  * mechanism uncertainty from posterior samples (rotation/Kagan angles,
    rotation-vector dispersion matrix),
  * for every station the predicted polarity and the posterior probability
    of the observed polarity,
  * a figure of the axis scatter and the rotation-angle distribution.

Events without inversion results are skipped.

Output per event (<data_dir>/events/<event_id>/inversion/):
    <event_id>_postprocess_report.txt     human-readable summary
    <event_id>_postprocess_summary.csv    parameter_name,value table
    <event_id>_station_predictions.csv    per-station predictions
    <event_id>_axes_uncertainty.png       uncertainty figure

Examples:
    python 05_postprocess.py --config config.yaml
    python 05_postprocess.py --config config.yaml --event 20020817.160627
"""

import csv
import traceback

import numpy as np

from fmtools.catalog import read_event_table
from fmtools.config import MissingInputError, add_event_selection_arguments, base_argument_parser, load_config, require, run
from fmtools.postprocess import (
    load_inversion,
    mechanism_uncertainty,
    plot_uncertainty,
    report_text,
    station_predictions,
    summary_rows,
)


def main():
    parser = base_argument_parser(__doc__)
    add_event_selection_arguments(parser)
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Do not make the uncertainty figure.",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    pp_cfg = cfg.get("postprocessing", {}) or {}

    events = read_event_table(cfg, args.event)

    done, skipped, failed = [], [], []
    for _, row in events.iterrows():
        event_id = row["event_id"]
        print("\n" + "=" * 70)
        print("Event {}".format(event_id))
        print("=" * 70)

        try:
            npz_path = require(
                cfg.inversion_file(event_id, "dc_grid_ppd.npz"),
                "Inversion results",
                "run 04_invert_focal_mechanism.py first",
            )
        except MissingInputError as exc:
            print("SKIP: {}".format(exc))
            skipped.append(event_id)
            continue

        try:
            inv = load_inversion(npz_path)
            rng = np.random.default_rng(int(pp_cfg.get("random_seed", 1234)))
            unc = mechanism_uncertainty(inv, int(pp_cfg.get("n_samples", 2000)), rng)
            pred = station_predictions(inv)

            report = report_text(inv, unc, pred)
            cfg.inversion_file(event_id, "postprocess_report.txt").write_text(report)
            print(report)

            with open(cfg.inversion_file(event_id, "postprocess_summary.csv"), "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["parameter_name", "value"])
                writer.writerows(summary_rows(inv, unc, pred))

            pred.to_csv(cfg.inversion_file(event_id, "station_predictions.csv"), index=False)

            if not args.no_plot:
                plot_file = cfg.inversion_file(event_id, "axes_uncertainty.png")
                plot_uncertainty(inv, unc, pred, plot_file)
                print("Uncertainty figure: {}".format(plot_file))

            print("Outputs written to {}".format(cfg.inversion_dir(event_id)))
            done.append(event_id)

        except Exception:
            traceback.print_exc()
            failed.append(event_id)

    print("\nDone. Post-processed: {}, skipped: {}, failed: {}".format(len(done), len(skipped), len(failed)))
    for label, ids in (("Skipped", skipped), ("Failed", failed)):
        if ids:
            print("{}: {}".format(label, ", ".join(ids)))


if __name__ == "__main__":
    run(main)
