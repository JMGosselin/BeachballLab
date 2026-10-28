"""
Reading the tutorial configuration file and building standard paths.

All scripts share one YAML config file. Relative paths in that file are
resolved relative to the directory containing the config file, so the
scripts can be run from anywhere.
"""

import argparse
import sys
from pathlib import Path

import yaml


REQUIRED_SECTIONS = [
    "project",
    "event_search",
    "stations",
    "waveforms",
    "preprocessing",
    "velocity_model",
    "picking",
    "inversion",
]


class Config(dict):
    """
    Dictionary holding the parsed YAML config, plus helpers for the
    standard on-disk layout of the tutorial:

        <data_dir>/
            catalog/events.csv          event list (step 1)
            catalog/events.xml          QuakeML
            catalog/events.pf           Pyrocko event file
            events/<event_id>/
                event.pf                Pyrocko event file
                raw/                    raw MiniSEED (step 2)
                stationxml/             station metadata (step 2)
                processed/              pre-processed MiniSEED (step 2)
                stations.csv            station geometry table (step 2)
                stations.pf             Pyrocko station file (step 2)
                station_map.png         station coverage plot (step 2)
                picks/<event_id>_markers.txt   Snuffler markers (step 3)
                inversion/                     inversion results (step 4)
    """

    def __init__(self, data, config_path):
        super().__init__(data)
        self.config_path = Path(config_path).resolve()
        self.base_dir = self.config_path.parent

    def resolve_path(self, path):
        path = Path(path).expanduser()
        if not path.is_absolute():
            path = self.base_dir / path
        return path

    # ------------------------------------------------------------------
    # Standard locations
    # ------------------------------------------------------------------
    @property
    def data_dir(self):
        return self.resolve_path(self["project"]["data_dir"])

    @property
    def catalog_dir(self):
        return self.data_dir / "catalog"

    @property
    def catalog_csv(self):
        return self.catalog_dir / "events.csv"

    def event_dir(self, event_id):
        return self.data_dir / "events" / event_id

    def raw_dir(self, event_id):
        return self.event_dir(event_id) / "raw"

    def stationxml_dir(self, event_id):
        return self.event_dir(event_id) / "stationxml"

    def processed_dir(self, event_id):
        return self.event_dir(event_id) / "processed"

    def stations_csv(self, event_id):
        return self.event_dir(event_id) / "stations.csv"

    def stations_pf(self, event_id):
        return self.event_dir(event_id) / "stations.pf"

    def event_pf(self, event_id):
        return self.event_dir(event_id) / "event.pf"

    def picks_dir(self, event_id):
        return self.event_dir(event_id) / "picks"

    def markers_file(self, event_id):
        return self.picks_dir(event_id) / "{}_markers.txt".format(event_id)

    def inversion_dir(self, event_id):
        return self.event_dir(event_id) / "inversion"

    def inversion_file(self, event_id, suffix):
        return self.inversion_dir(event_id) / "{}_{}".format(event_id, suffix)


class MissingInputError(FileNotFoundError):
    """
    A file that an earlier step should have produced does not exist.
    """


def require(path, what, hint):
    """
    Raise MissingInputError unless ``path`` exists (a directory must also
    be non-empty). ``hint`` tells the user which step creates it.
    """
    path = Path(path)
    ok = path.is_file() or (path.is_dir() and any(path.iterdir()))
    if not ok:
        raise MissingInputError("{} not found: {}\n    -> {}".format(what, path, hint))
    return path


def load_config(config_path):
    config_path = require(config_path, "Config file", "check the path given with --config")

    with open(config_path, "r") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError("Config file is empty or not a YAML mapping: {}".format(config_path))

    missing = [s for s in REQUIRED_SECTIONS if s not in data]
    if missing:
        raise ValueError(
            "Config file {} is missing section(s): {}".format(config_path, ", ".join(missing))
        )

    return Config(data, config_path)


def run(main):
    """
    Run a script's main() and turn missing inputs into a short error
    message instead of a traceback.
    """
    try:
        main()
    except MissingInputError as exc:
        sys.exit("ERROR: {}".format(exc))


def base_argument_parser(description):
    """
    Argument parser shared by all tutorial scripts.
    """
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-c",
        "--config",
        required=True,
        help="Path to the tutorial YAML config file (e.g. config.yaml).",
    )
    return parser


def add_event_selection_arguments(parser):
    parser.add_argument(
        "-e",
        "--event",
        action="append",
        default=None,
        metavar="EVENT_ID",
        help="Only process this event id (as listed in catalog/events.csv). "
        "Can be given several times. Default: all events in the catalog.",
    )
