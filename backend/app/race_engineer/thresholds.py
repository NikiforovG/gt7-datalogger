"""Detector thresholds, mirrored from the visual alert engine.

`frontend/src/lib/alerts.ts` THRESHOLDS drives the dashboard's colors and
alert strip. Voice must agree with what the driver sees, so the trigger
values here are the same numbers — keep them in sync (docs/internals/
race-engineer.md carries the table).

Voice adds two things the visual alerts do not need:

- a *clear* threshold below each trigger (hysteresis), so a value hovering on
  the limit doesn't produce a stream of "water temperature high" / silence;
- a persistence window, so one noisy sample never speaks.
"""

from __future__ import annotations

# --- fuel (laps of range remaining) -----------------------------------------
FUEL_LAPS_WARN = 3.0
FUEL_LAPS_CRITICAL = 1.5
# Announce a shortage only when the projection is short of the finish by more
# than this, so a rounding-level deficit stays quiet.
FUEL_SHORTFALL_MARGIN_LAPS = 0.25
# Consecutive laps the shortfall must persist before it is announced.
FUEL_SHORTFALL_MIN_LAPS = 2
# Re-announce a known shortage when it deepens by this much.
FUEL_SHORTFALL_ESCALATION_LAPS = 0.5

# --- engine ------------------------------------------------------------------
WATER_WARN = 110.0
WATER_WARN_CLEAR = 106.0
WATER_CRITICAL = 120.0
WATER_CRITICAL_CLEAR = 116.0

OIL_TEMP_WARN = 130.0
OIL_TEMP_WARN_CLEAR = 126.0
OIL_TEMP_CRITICAL = 140.0
OIL_TEMP_CRITICAL_CLEAR = 136.0

OIL_PRESSURE_MIN = 2.0  # bar
OIL_PRESSURE_CLEAR = 2.4
# Below idle-ish rpm the reading is meaningless (and is 0 with the engine off).
OIL_PRESSURE_MIN_RPM = 2000.0

TEMP_PERSISTENCE_S = 5.0
PRESSURE_PERSISTENCE_S = 1.5

# --- tires -------------------------------------------------------------------
TIRE_TEMP_WARN = 110.0
TIRE_TEMP_CLEAR = 104.0
TIRE_PERSISTENCE_S = 5.0
# Front/rear average difference worth mentioning after a lap.
TIRE_IMBALANCE_C = 8.0

# --- pace --------------------------------------------------------------------
# Laps averaged for the trend, and how many counted laps must exist first — a
# stint needs a settled reference before "your pace is dropping" means anything.
PACE_WINDOW_LAPS = 3
PACE_MIN_LAPS = 4
PACE_DROP_MS = 800.0  # average this far off the session best is a real slump
PACE_CLEAR_MS = 400.0  # ...and back inside this counts as recovered
PACE_ESCALATION_MS = 500.0  # further slippage worth mentioning again

# --- coaching ----------------------------------------------------------------
# Lockups/wheelspin are bucketed by track distance; occurrences of the same
# wheel inside one bucket across recent laps are what "repeated" means.
COACH_BUCKET_M = 120.0
COACH_WINDOW_LAPS = 3
COACH_MIN_OCCURRENCES = 3
COACH_LOCKUP_MAX_SLIP = 0.8  # severity gate: below this is a real lock
COACH_WHEELSPIN_MIN_SLIP = 1.25
# How far from a corner an event may be and still be named after it (a braking
# zone is long; naming a corner 300 m away would be wrong).
COACH_CORNER_WINDOW_M = 250.0
# Standalone braking-point coaching: how many recent laps must agree, and how
# far off the reference the mildest of them must still be. Wider than the
# descriptive clause's 5 m floor — this one tells a driver to change something.
BRAKE_WINDOW_LAPS = 2
BRAKE_POINT_MIN_M = 10.0

# Minimum time lost in one corner (vs the session best) before it is coached.
CORNER_LOSS_MIN_MS = 150.0
# What counts as a braking point — the pedal gate, how far before a corner it
# may begin, which corner an application belongs to — is not a threshold of
# the engineer's: it is processing/corner_metrics' definition, shared with
# the lap analysis document. Only what is worth SAYING is decided here.
# Differences smaller than these are inside lap-to-lap noise; mentioning them
# would send a driver chasing a braking marker they already hit.
CORNER_BRAKE_DIFF_MIN_M = 5.0
CORNER_APEX_SPEED_DIFF_MIN_KMH = 2.0
