"""
Velocity model handling with Pyrocko's cake ray tracer.

Used for predicted arrival times in Snuffler. The same model and functions
will later provide take-off angles for the focal-mechanism inversion.
"""

from pathlib import Path

from pyrocko import cake


def load_velocity_model(cfg):
    """
    Load either a cake built-in model (e.g. 'ak135-f-continental.m') or a
    .nd file given relative to the config file.
    """
    name = cfg["velocity_model"]["model"]
    path = cfg.resolve_path(name)
    if path.is_file():
        return cake.load_model(str(path))
    if Path(name).suffix == ".nd" or "/" in name:
        raise FileNotFoundError("Velocity model file not found: {}".format(path))
    return cake.load_model(name)


def phase_defs(spec):
    """
    'P,p' -> list of cake PhaseDef objects for the phases P and p.
    """
    defs = []
    for name in spec.split(","):
        name = name.strip()
        if name:
            defs.extend(cake.PhaseDef.classic(name))
    return defs


def first_arrival(mod, distance_km, source_depth_km, phases):
    """
    Earliest cake arrival for the given phase definitions, or None.
    """
    arrivals = mod.arrivals(
        [distance_km * 1000.0 * cake.m2d],
        phases=phases,
        zstart=source_depth_km * 1000.0,
    )
    if not arrivals:
        return None
    return min(arrivals, key=lambda a: a.t)
