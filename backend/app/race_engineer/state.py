"""Shared detector state: the packet clock and the per-lap context object."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.models import TelemetryPacket
from app.processing import alignment
from app.processing.corner_metrics import CornerMeasure, LapTrace, measure, windows
from app.processing.laps import CompletedLap
from app.processing.strategy import LapFuel

Samples = dict[str, list[float]]

# Same rule as the lap processor: time comes from the console's packet
# counter, never the wall clock. Pauses add no time (GT7 keeps streaming at
# 60 Hz while paused) and a dropped datagram widens one step instead of
# silently shrinking every persistence window.
TICK_SECONDS = 1 / 60
MAX_FRAME_GAP = 60


@dataclass(slots=True)
class PacketClock:
    """Monotonic seconds derived from packet ids."""

    now: float = 0.0
    _last_pid: int = -1

    def reset(self) -> None:
        self.now = 0.0
        self._last_pid = -1

    def advance(self, p: TelemetryPacket) -> float:
        gap = p.packet_id - self._last_pid if self._last_pid >= 0 else 1
        self._last_pid = p.packet_id
        # First packet, packet-id reset, or a discontinuity: count one frame.
        self.now += (gap if 1 <= gap <= MAX_FRAME_GAP else 1) * TICK_SECONDS
        return self.now


@dataclass(slots=True)
class LapOnAxis:
    """A lap placed on the coaching reference's distance axis.

    A lap's own `dist` is integrated from its own speed, so two laps reach
    the same metre mark at different places on the road — a median 2.9 m
    apart and 64 m at worst over real lap pairs (see processing/alignment).
    Coaching compares braking points to within 5 m, so it compares them on
    ONE axis: the reference lap's, which every other lap is projected onto
    by where it was. The events move with the samples.

    `aligned` is False for a lap that could not be placed with confidence
    (and for every lap while there is no reference): it keeps its own
    distance, which is what every comparison used before.
    """

    # The reference this lap was placed against, held so that a change of
    # reference is noticed by identity and the lap placed again.
    reference: Samples | None
    samples: Samples
    events: list[dict[str, Any]]
    aligned: bool
    _measures: dict[int, CornerMeasure] | None = None
    _measured_for: list[dict[str, float | int | str]] | None = None

    def measures(
        self, corners: list[dict[str, float | int | str]]
    ) -> dict[int, CornerMeasure]:
        """What the lap did at each of `corners`, by corner number. Worked
        out once per set of corners: a millisecond or two a lap, but asked
        for by every corner of every comparison."""
        if self._measures is None or self._measured_for is not corners:
            self._measures = measure(LapTrace(self.samples), windows(list(corners)))
            self._measured_for = corners
        return self._measures


def place_on_axis(
    samples: Samples,
    events: list[dict[str, Any]],
    reference: Samples | None,
) -> LapOnAxis:
    """`samples` on `reference`'s distance axis. Tens of milliseconds for a
    full lap: the live path runs it on a worker thread (the manager's
    `prepare_lap`), the replay is on one already."""
    if reference is None or samples is reference:
        return LapOnAxis(reference, samples, events, aligned=samples is reference)
    path = alignment.ReferencePath(reference) if reference.get("pos_x") else None
    moved = alignment.align_to_reference(samples, path) if path is not None else None
    if moved is None:
        return LapOnAxis(reference, samples, events, aligned=False)
    own = samples.get("dist") or []
    return LapOnAxis(
        reference,
        moved,
        alignment.remap_events(events, own, moved["dist"]),
        aligned=True,
    )


@dataclass(slots=True)
class LapRecord:
    """A completed lap as the detectors need it (kept across sessions)."""

    number: int
    time_ms: int
    car_id: int
    fuel_consumed: float
    counts_for_best: bool
    # Lap numbers repeat across sessions and the history deliberately spans
    # them (for the fuel model), so re-flagging a partial lap has to know
    # which session's lap 1 it means.
    session_seq: int = 0
    invalidated_best: bool = False
    events: list[dict[str, Any]] = field(default_factory=list)
    samples: dict[str, list[float]] = field(default_factory=dict)
    # The lap on the coaching reference's axis, once something has asked for
    # it (EngineerContext.on_axis). Stale as soon as the reference changes.
    axis: LapOnAxis | None = None

    @classmethod
    def from_lap(cls, lap: CompletedLap) -> LapRecord:
        return cls(
            number=lap.number,
            time_ms=lap.time_ms,
            car_id=lap.car_id,
            fuel_consumed=lap.fuel_consumed,
            counts_for_best=lap.counts_for_best,
            invalidated_best=lap.invalidated_best,
            events=lap.events,
            samples=lap.samples,
        )

    def as_fuel(self) -> LapFuel:
        return LapFuel(
            number=self.number,
            time_ms=self.time_ms,
            fuel_consumed=self.fuel_consumed,
            car_id=self.car_id,
        )


@dataclass(slots=True)
class EngineerContext:
    """Everything detectors may read; owned and updated by the manager.

    `laps` deliberately survives session boundaries (newest first): a race
    restart opens a new session, and dropping the stint's laps would blank the
    fuel model exactly when it matters. Same-car filtering happens in the
    projection, as it does on the frontend.
    """

    now: float = 0.0
    session_id: int | None = None
    session_seq: int = 0  # bumped on every session; part of dedupe keys
    # "metric" or "imperial" — spoken units for distances and speeds. The
    # browser's own km/h-vs-mph toggle is per-device and can't reach text the
    # server has already worded, so this is a server setting.
    units: str = "metric"
    track_name: str = ""
    car_id: int = 0
    best_lap_ms: int | None = None
    prev_best_ms: int | None = None
    # Most recent packet, so lap-boundary detectors can read live values
    # (fuel level, race distance) without the manager passing them along.
    packet: TelemetryPacket | None = None
    laps: list[LapRecord] = field(default_factory=list)
    # Reference lap for coaching: the session-best lap's samples and the
    # corners detected on it (empty until a best lap exists).
    reference: dict[str, list[float]] | None = None
    corners: list[dict[str, float | int | str]] = field(default_factory=list)
    # False until several laps agree on the track's distance. Everything that
    # compares one lap against another by position — braking points, corner
    # losses, where a lockup happened — is meaningless before then, because a
    # lap the logger only half-saw has its distance axis anchored elsewhere.
    span_confirmed: bool = False
    # The reference lap on its own axis — it is rarely one of `laps` (the
    # history is capped, and the replay keeps no record of it at all).
    _reference_axis: LapOnAxis | None = None

    def on_axis(self, rec: LapRecord) -> LapOnAxis:
        """`rec` on the reference's distance axis, placed once per reference.

        A lap from an earlier session stays on its own axis: the history
        spans sessions for the fuel model's sake, and a lap driven before a
        restart may not even be of this circuit.
        """
        ref = self.reference if rec.session_seq == self.session_seq else None
        view = rec.axis
        if view is None or view.reference is not ref:
            view = place_on_axis(rec.samples, rec.events, ref)
            rec.axis = view
        return view

    def reference_measures(self) -> dict[int, CornerMeasure]:
        """What the reference lap itself did at each corner."""
        ref = self.reference
        if ref is None:
            return {}
        view = self._reference_axis
        if view is None or view.reference is not ref:
            view = place_on_axis(ref, [], ref)
            self._reference_axis = view
        return view.measures(self.corners)

    def corner_at(self, dist_m: float) -> int | None:
        """Corner number containing a track distance, if any.

        A start/finish corner reports entry_dist > exit_dist because its
        extent wraps past the line — the containment test has to wrap too.
        """
        for corner in self.corners:
            entry = float(corner["entry_dist"])
            exit_ = float(corner["exit_dist"])
            inside = (
                entry <= dist_m <= exit_
                if entry <= exit_
                else (dist_m >= entry or dist_m <= exit_)
            )
            if inside:
                return int(corner["n"])
        return None

    def corner_name(self, number: int | None) -> str:
        """A corner's hand-given name, if this circuit's corners were labelled.

        Only authored corners carry one — detection has nothing to name a
        corner after (#48).
        """
        if number is None:
            return ""
        for corner in self.corners:
            if int(corner["n"]) == number:
                return str(corner.get("name") or "")
        return ""

    def corner_ahead(self, dist_m: float, window_m: float) -> int | None:
        """The corner a driver at this distance is braking for, if any.

        Braking events land *before* the corner they belong to, so the corner
        containing them is usually None — the useful answer is the next one
        within a braking zone's reach.
        """
        best: int | None = None
        best_gap = window_m
        for corner in self.corners:
            gap = float(corner["entry_dist"]) - dist_m
            if 0 <= gap < best_gap:
                best_gap, best = gap, int(corner["n"])
        return best

    def corner_behind(self, dist_m: float, window_m: float) -> int | None:
        """The corner just exited (wheelspin happens on corner exit)."""
        best: int | None = None
        best_gap = window_m
        for corner in self.corners:
            gap = dist_m - float(corner["exit_dist"])
            if 0 <= gap < best_gap:
                best_gap, best = gap, int(corner["n"])
        return best
