#!/usr/bin/env python3
"""
Step 3: pick P-wave first-motion polarities in Snuffler.

Opens Snuffler for each event with the processed waveforms, station
locations, the event and predicted arrival markers (P_pred, S_pred).
When you close the Snuffler window, your picks are saved to

    <data_dir>/events/<event_id>/picks/<event_id>_markers.txt

and re-loaded the next time you open the same event.

Picking in Snuffler (see README for details):
    double-click, then click   create a marker at the P onset
    F1                         make it a "P" phase marker
    Up / Down arrow            polarity UP (compression) / DOWN (dilatation)
    Shift + Up / Down          polarity unclear
    1 / 2 / 3                  confidence class (clear / OK / uncertain)
    Backspace                  delete the selected marker

Examples:
    python 03_pick_polarities.py --config config.yaml
    python 03_pick_polarities.py --config config.yaml --event 20020817.160627
    python 03_pick_polarities.py --config config.yaml --summary
"""

from fmtools.catalog import read_event_table, row_to_pyrocko_event
from fmtools.config import MissingInputError, add_event_selection_arguments, base_argument_parser, load_config, run
from fmtools.picking import polarity_table, read_polarity_picks, run_snuffler, save_markers
from fmtools.waveforms import read_station_table


def print_summary(event_id, picks):
    if picks.empty:
        print("Event {}: no polarity picks.".format(event_id))
        return
    counts = picks["polarity"].value_counts()
    print("Event {}: {} pick(s)  [U: {}, D: {}, unclear/none: {}]".format(
        event_id, len(picks), counts.get("U", 0), counts.get("D", 0),
        len(picks) - counts.get("U", 0) - counts.get("D", 0)))
    print(picks.to_string(index=False))


def ask(question):
    try:
        answer = input(question).strip().lower()
    except EOFError:
        return "q"
    return answer


def main():
    parser = base_argument_parser(__doc__)
    add_event_selection_arguments(parser)
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Do not open Snuffler; print the saved polarity picks instead.",
    )
    parser.add_argument(
        "--unpicked",
        action="store_true",
        help="Only open events that do not have a marker file yet.",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)

    events = read_event_table(cfg, args.event)

    if args.summary:
        for _, row in events.iterrows():
            try:
                print_summary(row["event_id"], read_polarity_picks(cfg, row["event_id"]))
            except MissingInputError as exc:
                print("Event {}: {}".format(row["event_id"], exc))
            print()
        return

    if args.unpicked:
        events = events[[not cfg.markers_file(e).is_file() for e in events["event_id"]]]
        if events.empty:
            print("All selected events already have picks.")
            return

    for i, (_, row) in enumerate(events.iterrows()):
        event_id = row["event_id"]

        if len(events) > 1:
            answer = ask("\n[{}/{}] Open event {} in Snuffler? [Y]es / [s]kip / [q]uit: ".format(
                i + 1, len(events), event_id))
            if answer.startswith("q"):
                break
            if answer.startswith("s"):
                continue

        event = row_to_pyrocko_event(row)
        try:
            stations = read_station_table(cfg, event_id)
            markers = run_snuffler(cfg, event, stations)
        except MissingInputError as exc:
            print("SKIP event {}: {}".format(event_id, exc))
            continue

        path = save_markers(cfg, event_id, markers)
        print("\nSaved markers to {}".format(path))
        print_summary(event_id, polarity_table(cfg, markers))


if __name__ == "__main__":
    run(main)
