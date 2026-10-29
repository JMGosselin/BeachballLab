# BeachballLab

**Focal mechanisms from P-wave polarities: a hands-on tutorial.**

This tutorial walks you through estimating an earthquake focal mechanism
from P-wave first-motion polarities, starting from public data:

1. **Search** an earthquake catalogue for events in a region.
2. **Download** and pre-process seismic waveforms for each event.
3. **Pick** P-wave first-motion polarities interactively in
   [Pyrocko Snuffler](https://pyrocko.org/docs/current/apps/snuffler/).
4. **Invert** the polarities for a focal mechanism: take-off angles from a
   velocity model, a probabilistic grid search, and a fuzzy beachball plot.
5. **Post-process** the result: stress axes, faulting style, uncertainty and
   how well each station fits.

Every script reads the **same configuration file** (`config.yaml`). To work
on your own region or events, copy `config.yaml`, edit it, and point the
scripts at your copy.

The worked example is the **M4.5 earthquake of 2002-08-17 16:06:27 UTC,
39 km ESE of Merritt, British Columbia** (49.960°N, 120.290°W, 10 km depth;
USGS event `usp000ba34`).

---

## Installation

```bash
git clone https://github.com/JMGosselin/BeachballLab.git
cd BeachballLab
conda env create -f environment.yml
conda activate beachballlab
```

All commands below are run from the `BeachballLab` folder.

---

## Folder layout

```
config.yaml                 all parameters for every step
01_search_events.py         step 1: event search
02_download_waveforms.py    step 2: waveform download + pre-processing
03_pick_polarities.py       step 3: polarity picking in Snuffler
04_invert_focal_mechanism.py  step 4: focal-mechanism inversion + plot
05_postprocess.py           step 5: axes, uncertainty, station fit
fmtools/                    shared Python code used by the scripts
examples/                   reference picks for the worked example
data/                       created by the scripts (not in git)
```

### Where the outputs are saved

All outputs go into the data folder set by `project.data_dir` in the config
(`./data` by default). A relative path is taken relative to the folder that
contains the config file, not the folder you run the script from. Each
event gets its own folder, `data/events/<event_id>/`, where `<event_id>` is
built from the origin time, e.g. `20020817.160627`.

```
data/
├── catalog/                                  STEP 1
│   ├── events.csv                            event list used by all later steps
│   ├── events.xml                            full catalogue (QuakeML)
│   └── events.pf                             events for Pyrocko
└── events/
    └── <event_id>/
        ├── raw/NET.STA.LOC.CHA.mseed         STEP 2: raw waveforms
        ├── stationxml/NET.STA.xml            STEP 2: station metadata + responses
        ├── processed/NET.STA.LOC.CHA.mseed   STEP 2: response removed, ZNE, Z up
        ├── stations.csv                      STEP 2: distance, azimuth, back-azimuth
        ├── stations.pf                       STEP 2: stations for Snuffler
        ├── event.pf                          STEP 2: event for Snuffler
        ├── station_map.png                   STEP 2: station coverage map
        ├── picks/                            STEP 3
        │   ├── <event_id>_markers.txt        your Snuffler picks
        │   └── <event_id>_markers.txt.bak    previous version of the picks
        └── inversion/
            ├── <event_id>_polarities.csv         STEP 4: picks + azimuth + take-off angle
            ├── <event_id>_dc_grid_ppd.npz        STEP 4: full grid and posterior
            ├── <event_id>_inversion_summary.txt  STEP 4: solutions + station fit
            ├── <event_id>_fuzzy_beachball.png    STEP 4: fuzzy beachball + polarities
            ├── <event_id>_postprocess_report.txt STEP 5: readable report
            ├── <event_id>_postprocess_summary.csv STEP 5: same numbers as a table
            ├── <event_id>_station_predictions.csv STEP 5: per-station fit
            └── <event_id>_axes_uncertainty.png   STEP 5: P/T/N scatter + Kagan angles
```

For the example event, the fuzzy beachball is therefore
`data/events/20020817.160627/inversion/20020817.160627_fuzzy_beachball.png`.
The scripts also print the path of every file they write.

Each script checks that the files it needs from earlier steps exist. If
something is missing, that event is skipped with a message telling you
which step to run.

---

## Step 1: search for events

```bash
python 01_search_events.py --config config.yaml
```

Relevant config section: `event_search` (data centre, time window, region,
magnitude and depth limits). The search area can be a circle
(`region_type: circle`) or a latitude/longitude box (`region_type: box`).

**Output** (in `data/catalog/`): `events.csv`, `events.xml` (QuakeML) and
`events.pf` (Pyrocko format). **`events.csv` drives the following steps.**
To skip an event, delete its row from the file.

For the example config you should get one event:

```
       event_id              event_datetime  event_lat  event_lon  event_depth  event_mag event_mag_type
20020817.160627 2002-08-17T16:06:27.000000Z      49.96    -120.29         10.0        4.5             mb
```

---

## Step 2: download and pre-process waveforms

```bash
python 02_download_waveforms.py --config config.yaml
```

Relevant config sections: `stations`, `waveforms`, `preprocessing`.

For every event the script:

1. finds all stations between `min_radius_km` and `max_radius_km` from the
   epicentre at the configured data centres (`providers`), choosing one
   channel group per station following `channel_priorities` (e.g. HH before
   BH before EH);
2. downloads raw MiniSEED (`raw/`) and station metadata (`stationxml/`);
3. removes the instrument response (to velocity by default), rotates 1/2
   components to N/E, and checks that every vertical component is
   **positive up**;
4. writes processed data (`processed/`), a station table (`stations.csv`
   with distance, azimuth and back-azimuth) and a coverage map
   (`station_map.png`).

**Output** (in `data/events/<event_id>/`):

| File or folder | Contents |
| --- | --- |
| `raw/` | raw MiniSEED, one file per channel (`NET.STA.LOC.CHA.mseed`) |
| `stationxml/` | station metadata and instrument responses (`NET.STA.xml`) |
| `processed/` | pre-processed MiniSEED, read by Snuffler in step 3 |
| `stations.csv` | station table with distance, azimuth and back-azimuth |
| `stations.pf`, `event.pf` | station and event files for Snuffler |
| `station_map.png` | map of the stations around the event |

Useful options:

```bash
# one event only
python 02_download_waveforms.py --config config.yaml --event 20020817.160627
# re-run only the pre-processing (e.g. after changing the preprocessing section)
python 02_download_waveforms.py --config config.yaml --skip-download
```

Stations that were already downloaded are skipped on a re-run unless
`waveforms.overwrite: true`. The data centres in `providers` are queried in
order and a station is only taken from the first one that has it. For the
example event, Natural Resources Canada supplies the Canadian (CN, PO, C8)
stations and EarthScope the US (UW) stations, giving 42 stations at
86–300 km. NRCan allows only one request every 5 seconds
(`request_interval_s`), so the download takes a few minutes.

> **Why the pre-processing matters for polarities.** A polarity is only
> meaningful if "up" on the seismogram really is ground motion up. Some
> instruments have a negative gain (an inverted signal), which is corrected
> by removing the instrument response. Others are registered with a dip of
> +90° (sensor pointing down), and the script flips those. Such stations are
> reported in the output, e.g. `[UW.PGW..EHZ has dip +90; flipped to Z-up]`.

> **Look at `station_map.png`.** A focal mechanism is only well constrained
> when stations surround the event. The script prints the largest azimuthal
> gap; for the example it is about 100° (few stations to the east). Try
> increasing `max_radius_km` and see how the coverage changes.

---

## Step 3: pick polarities in Snuffler

```bash
python 03_pick_polarities.py --config config.yaml
```

Relevant config sections: `picking`, `velocity_model`.

Snuffler opens with the vertical-component traces sorted by distance, the
event, and markers at the **predicted** P and S arrival times
(`P_pred`, `S_pred`, computed with the velocity model). The predicted times
are only a guide: the real onset can be a few seconds earlier or later.

### Picking a polarity

| Action | Key / mouse |
| --- | --- |
| Create a marker at the P onset | **double-click** on the trace, then click once to drop it |
| Turn it into a **P** phase marker | `F1` |
| Polarity **up** (compression) | `↑` (press again to remove) |
| Polarity **down** (dilatation) | `↓` (press again to remove) |
| Onset visible but polarity unclear | `Shift+↑` or `Shift+↓` |
| Pick confidence class | `1` clear, `2` OK, `3` uncertain (no key = `0`) |
| Delete the selected marker(s) | `Backspace` |
| Zoom / pan in time | drag with the left mouse button: up/down zooms, left/right pans |
| Scroll through the traces | mouse wheel, or `Page Down` / `Page Up` |
| Show more / fewer traces | `Ctrl` + mouse wheel |
| Filter the display | Highpass / Lowpass sliders in the *Main Controls* panel |

The arrow keys and number keys act on the **selected** marker (it is
highlighted). Click on a marker to select it. The polarity keys only work
on phase markers, so press `F1` before `↑`/`↓`.

Tips:

* Zoom in strongly around the onset. The first motion is the very first
  swing away from the noise, not the largest peak.
* Use filtering with care. Strong filters can create a small precursor
  swing that has the opposite sign to the true first motion. If an onset
  only becomes clear after heavy filtering, give it a low confidence class.
* Do not set a polarity on the `P_pred` markers. Make your own `P` marker.
* The confidence classes are converted to numbers using
  `picking.confidence_by_kind` in the config (by default `1` = 0.95,
  `2` = 0.75, `3` = 0.55, and a marker left at `0` gets 0.80). The
  inversion uses them to weight each polarity. Classes not listed there
  get 0.5, with a warning in step 4.

### Saving

Close the Snuffler window when you are done. Your picks are saved
automatically to

```
data/events/<event_id>/picks/<event_id>_markers.txt
```

and a summary is printed. Running the script again re-opens the event
**with your previous picks loaded**, so you can continue or correct them.
The previous file is kept as `<event_id>_markers.txt.bak` in the same
`picks/` folder.

Useful options:

```bash
# one event only
python 03_pick_polarities.py --config config.yaml --event 20020817.160627
# only events that have not been picked yet
python 03_pick_polarities.py --config config.yaml --unpicked
# print the saved picks without opening Snuffler
python 03_pick_polarities.py --config config.yaml --summary
```

With several events, the script asks before opening each one
(`[Y]es / [s]kip / [q]uit`).

### Reference picks

`examples/20020817.160627_markers.txt` holds a set of picks for the worked
example (33 P picks: 20 up, 12 down, 1 unclear). Make your own picks first,
then compare. To run steps 4 and 5 with the reference picks instead of
yours, copy the file into place (this replaces your own picks, so back them
up first):

```bash
mkdir -p data/events/20020817.160627/picks
cp examples/20020817.160627_markers.txt data/events/20020817.160627/picks/
```

The results quoted in steps 4 and 5 below were obtained with these picks.

---

## Step 4: invert for the focal mechanism

```bash
python 04_invert_focal_mechanism.py --config config.yaml
```

Relevant config sections: `inversion`, `plotting`, `velocity_model`.

Only events with a marker file from step 3 are inverted, and only if they
have at least `inversion.min_polarities` clear up/down polarities (picks
with an unclear polarity are ignored).

For every event the script:

1. combines your P picks with the station table and computes each ray's
   **take-off angle** with the velocity model (the earliest of the phases in
   `takeoff_phases`, by default direct `p` or `P`, including the Pn head
   wave). Together with the station azimuth this tells us where the ray
   leaves the focal sphere. The table is saved as
   `inversion/<event_id>_polarities.csv`;
2. evaluates every double-couple mechanism on a strike/dip/rake grid
   (about 100,000 mechanisms with the default steps). For each one it
   predicts the P polarity at every station and computes how likely the
   observed polarities are. Picks with low confidence (`sigma_max`) count
   less than clear ones (`sigma_min`), and `omega` allows for a few plain
   wrong polarities. The result is a probability (posterior) for every
   mechanism, saved in `<event_id>_dc_grid_ppd.npz`;
3. writes a summary (`<event_id>_inversion_summary.txt`) with the
   maximum-posterior and posterior-mean ("centroid") solutions, both nodal
   planes, and a station table showing which polarities do **not** fit;
4. plots a **fuzzy beachball** (`<event_id>_fuzzy_beachball.png`). The
   shading is made from `plotting.n_posterior_draws` mechanisms drawn from
   the posterior: solid red/white where all likely mechanisms agree,
   pale where they disagree. The black lines are the nodal planes of the
   preferred solution (`inversion.preferred_solution`), and the symbols are
   your polarities (▲ up, ▽ down; larger = more confident) in a
   lower-hemisphere equal-area projection.

**Output** (in `data/events/<event_id>/inversion/`):

| File | Contents |
| --- | --- |
| `<event_id>_polarities.csv` | your picks with azimuth, take-off angle and confidence |
| `<event_id>_dc_grid_ppd.npz` | the full strike/dip/rake grid and posterior (read by step 5) |
| `<event_id>_inversion_summary.txt` | solutions and per-station fit (also printed to the screen) |
| `<event_id>_fuzzy_beachball.png` | fuzzy beachball with your polarities |

For the example event you should get a solution close to strike/dip/rake
292/78/−161 (other plane 198/71/−13), with 28 of 32 polarities fitting.

Things to try:

* Look at the misfit stations in the summary, check those picks in
  Snuffler, and re-run the inversion.
* Look at the `takeoff_angle` column of `<event_id>_polarities.csv`.
  Beyond about 130 km the first arrival is the Pn head wave, which leaves
  the source at the same take-off angle (about 46° in this model), so all
  these stations plot on one ring. Only the two closest stations (PNT and
  HOPB, direct upgoing P) sample other parts of the focal sphere. How does
  the solution change if you remove them?
* Change the velocity model or the event depth (`event_depth` in
  `data/catalog/events.csv`), re-run step 4 and see how the take-off
  angles and the mechanism change.
* Use `--no-plot` to skip the figure.

---

## Step 5: post-process the result

```bash
python 05_postprocess.py --config config.yaml
```

Relevant config section: `postprocessing`. Events without inversion results
from step 4 are skipped. Re-run step 5 whenever you re-run step 4.

The script reads the posterior from step 4
(`<event_id>_dc_grid_ppd.npz`) and writes its results next to it.

**Output** (in `data/events/<event_id>/inversion/`):

| File | Contents |
| --- | --- |
| `<event_id>_postprocess_report.txt` | readable report (also printed to the screen) |
| `<event_id>_postprocess_summary.csv` | the same numbers as a `parameter_name,value` table |
| `<event_id>_station_predictions.csv` | predicted polarity and `prob_obs` for every station |
| `<event_id>_axes_uncertainty.png` | P/T/N axis scatter and Kagan-angle histogram |

**Principal axes and faulting style.** The P (compression), T (tension) and
N (null) axes of the preferred mechanism are given as trend and plunge. The
faulting style is classified from their plunges with the World Stress Map
scheme (Zoback, 1992): normal (NF), normal/strike-slip (NS), strike-slip
(SS), thrust/strike-slip (TS), thrust (TF) or unclassified (U). The report
also lists which fraction of the posterior samples falls in each class,
which is a more honest answer than a single label.

**Uncertainty.** `postprocessing.n_samples` mechanisms are drawn from the
posterior. Any double couple can be turned into any other by one rotation;
the angle of that rotation (the *Kagan angle*, 0–120°) measures how
different two mechanisms are. The report gives the median, 68% and 95%
values of the Kagan angle between the samples and the preferred solution,
and `sigma`, the root-mean-square rotation (from the rotation-vector
dispersion matrix, RVDM). For comparison, a completely unconstrained
mechanism gives sigma ≈ 78°.

**Station fit.** For every polarity the report lists the polarity and
amplitude predicted by the preferred solution and `prob_obs`, the posterior
probability of the polarity you observed. Stations that do not match are
marked `misfit`; matching stations with a small amplitude are marked
`near nodal plane`, since their polarity is sensitive to small changes in
the mechanism.

**The figure.** Left: the P, T and N axes of posterior samples on a
lower-hemisphere stereonet, with the preferred axes and nodal planes. Tight
clusters mean well-resolved axes. Right: the histogram of Kagan angles.

For the example event you should find P 156/22, T 64/4, faulting style SS
(strike-slip), a median Kagan angle of about 18° and sigma ≈ 27°.

Things to look at:

* GOBB has `prob_obs` close to 0: almost no plausible mechanism fits it,
  although its neighbours on the stereonet are all up. Go back to Snuffler
  and check this pick. A station that is consistently wrong may have
  reversed polarity.
* LLLB, CMW and BLN also misfit, but with `prob_obs` of 0.2–0.5: they lie
  close to a nodal plane, where small changes in the mechanism flip the
  predicted polarity.
* PNT is one of only two stations off the Pn ring (see step 4), so a single
  pick there carries a lot of weight. Flip its polarity in the marker file,
  re-run steps 4 and 5, and compare the mechanism and faulting style.
* The Kagan-angle histogram has a small second group near 80° (about 5% of
  the samples). Which stations would you need to rule out these
  solutions?
* About 80% of the posterior samples are strike-slip. How confidently can
  you name the faulting style of this event?

---

## Using your own events

1. Copy `config.yaml` to e.g. `my_region.yaml`.
2. Change `project.data_dir` so the new data go to a separate folder.
3. Edit `event_search` (time window, region, magnitudes), and the station
   distance range and `providers` in `stations`.
4. Run the five steps with `--config my_region.yaml`.

FDSN data centres can be given as ObsPy short names (`EARTHSCOPE`, `USGS`,
`ISC`, `GEOFON`, ...) or full URLs (e.g.
`https://earthquakescanada.nrcan.gc.ca` for Natural Resources Canada).

To use your own velocity model, set `velocity_model.model` to a Pyrocko
`.nd` file (path relative to the config file). See the
[cake documentation](https://pyrocko.org/docs/current/apps/cake/manual.html)
for the format.

---

## Recommendations

### For students

* **Pick first, then compare.** Make your own picks before you look at
  `examples/`. Compare the two sets of polarities station by station, then
  compare the two mechanisms.
* **Test how robust the solution is.** Remove one station at a time,
  especially PNT or HOPB, the only two off the Pn ring. Watch how much the
  mechanism and the Kagan-angle spread change.
* **Try different source depths.** Re-run steps 4 and 5 with the depth at
  5, 10 and 15 km. Which stations move on the focal sphere, and does the
  faulting style change?
* **Try other events.** Run the workflow on another event in the region.
  See whether the faulting style matches this one and fits what you would
  expect for the regional tectonics.
* **Compare with a published solution.** Look for a moment tensor or
  focal mechanism for the same event in a catalogue, if one exists.
  Explain any difference.

### Ideas for improving the lab

* **Interactive beachball fitter.** Sliders for strike, dip and rake that
  redraw the beachball over the observed polarities, so students fit a
  mechanism by hand before running the inversion.
* **Basemap on the station map.** Coastlines, borders and topography behind
  the stations and the event, for example with Cartopy or PyGMT.
* **Run-all script.** A script that runs steps 1, 2, 4 and 5 for every
  event in the catalogue, skipping the interactive picking.
* **Take-off angle uncertainty.** Sample the source depth and the velocity
  model inside the inversion, rather than treating the take-off angles as
  exact.
* **Regional velocity model.** A 1-D model for southern British Columbia in
  place of the global ak135 model.
* **S/P amplitude ratios.** Add amplitude ratios as extra data, as HASH
  does, to help constrain mechanisms with few or clustered polarities.
* **Station polarity checks.** Flag stations whose polarity disagrees with
  the solution across many events. This would catch reversed stations like
  the one GOBB may be.
* **Jupyter notebook version.** A notebook that steps through the same
  workflow for classroom demos.

---

## Licence

BeachballLab is released under the GNU General Public License v3.0 (see
`LICENSE`).
