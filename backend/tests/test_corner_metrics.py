"""What a lap did at a corner (app.processing.corner_metrics).

The laps here are one-dimensional — a distance axis and the pedals along it —
because every question asked is one of place along the lap: which corner a
brake application belongs to, where the car turned in, where the throttle
came back.
"""

import pytest

from app.processing.corner_metrics import (
    BRAKE_SEARCH_M,
    LapTrace,
    brake_point_delta,
    measure,
    min_speed_delta,
    windows,
)

SPEED_MPS = 50.0


def lap(
    length: int = 2000,
    brake: list[tuple[float, float, float]] | None = None,
    lift: list[tuple[float, float]] | None = None,
    slow: list[tuple[float, float, float]] | None = None,
    yaw: list[tuple[float, float, float]] | None = None,
    part_throttle: list[tuple[float, float, float]] | None = None,
) -> dict[str, list[float]]:
    """A lap in 1 m steps at 50 m/s.

    `brake` is (from, to, percent) applications; `lift` (from, to) stretches
    off the throttle; `slow` (from, to, km/h) stretches below the 180 km/h the
    rest of the lap runs at; `yaw` (from, to, rad/s) stretches of turning.
    """
    dist = [float(d) for d in range(length)]

    def level(spans, default, d):  # noqa: ANN001, ANN202 - test helper
        for lo, hi, value in spans or []:
            if lo <= d < hi:
                return value
        return default

    throttle = []
    for d in dist:
        off = any(lo <= d < hi for lo, hi in lift or []) or any(
            lo <= d < hi for lo, hi, _ in brake or []
        )
        throttle.append(level(part_throttle, 0.0 if off else 100.0, d))
    return {
        "dist": dist,
        "t": [d / SPEED_MPS for d in dist],
        "speed": [level(slow, 180.0, d) for d in dist],
        "brake": [level(brake, 0.0, d) for d in dist],
        "throttle": throttle,
        "yaw_rate": [level(yaw, 0.0, d) for d in dist],
        "gear": [3.0 if level(slow, 180.0, d) < 180.0 else 5.0 for d in dist],
    }


def corner(n: int, apex: float, half: float = 75.0) -> dict[str, float | int]:
    return {"n": n, "entry_dist": apex - half, "apex_dist": apex, "exit_dist": apex + half}


def measured(samples: dict[str, list[float]], corners: list[dict]) -> dict:
    return measure(LapTrace(samples), windows(corners))


def test_a_braking_zone_belongs_to_the_corner_it_slowed_the_car_for() -> None:
    """Through a sequence the 250 m before the second corner's entry holds the
    first corner's braking. Each application goes to one corner only."""
    corners = [corner(1, 600.0), corner(2, 760.0)]
    samples = lap(
        brake=[(480.0, 580.0, 100.0), (700.0, 740.0, 60.0)],
        slow=[(580.0, 640.0, 90.0), (740.0, 800.0, 110.0)],
    )
    # What the corner's own window holds of the other's braking:
    assert 480.0 > corners[1]["entry_dist"] - BRAKE_SEARCH_M
    m = measured(samples, corners)
    assert m[1].brake_on == 480.0
    assert m[1].brake_off == 579.0
    assert m[2].brake_on == 700.0
    assert m[2].brake_peak == 60.0


def test_braking_that_begins_after_the_entry_marker_is_still_found() -> None:
    """An authored corner's window is its apex ± 75 m, not where the driver
    turned in: in a tight sequence the brake goes on inside it."""
    m = measured(lap(brake=[(540.0, 590.0, 80.0)]), [corner(1, 600.0)])
    assert m[1].brake_on == 540.0


def test_a_brake_application_far_up_the_straight_belongs_to_no_corner() -> None:
    corners = [corner(1, 1500.0)]
    far = corners[0]["entry_dist"] - BRAKE_SEARCH_M - 100.0
    m = measured(lap(brake=[(far, far + 40.0, 100.0)]), corners)
    assert m[1].brake_on is None
    assert m[1].brake_off is None


def test_the_application_that_took_the_most_speed_off_is_the_zone() -> None:
    """A dab on the way in, then the braking itself."""
    samples = lap(
        brake=[(380.0, 400.0, 30.0), (480.0, 580.0, 100.0)],
        slow=[(390.0, 480.0, 175.0), (570.0, 640.0, 90.0)],
    )
    m = measured(samples, [corner(1, 600.0)])
    assert m[1].brake_on == 480.0
    assert m[1].brake_on_speed == 180.0
    assert m[1].brake_off_speed == 90.0


def test_a_pedal_dipping_under_the_gate_does_not_split_the_zone() -> None:
    m = measured(
        lap(brake=[(480.0, 520.0, 100.0), (525.0, 580.0, 70.0)]), [corner(1, 600.0)]
    )
    assert m[1].brake_on == 480.0
    assert m[1].brake_off == 579.0


def test_a_brush_of_the_pedal_is_not_a_braking_zone() -> None:
    # Three samples at 50 m/s is 0.04 s.
    m = measured(lap(brake=[(500.0, 503.0, 100.0)]), [corner(1, 600.0)])
    assert m[1].brake_on is None


def test_a_brake_already_on_at_the_line_has_no_onset_in_this_lap() -> None:
    m = measured(lap(brake=[(0.0, 60.0, 100.0)]), [corner(1, 80.0)])
    assert m[1].brake_on is None
    assert m[1].brake_off == 59.0
    assert m[1].brake_peak == 100.0


def test_turn_in_and_the_brake_carried_past_it() -> None:
    samples = lap(
        brake=[(480.0, 575.0, 100.0)],
        slow=[(560.0, 640.0, 90.0)],
        yaw=[(560.0, 650.0, 0.6)],
    )
    m = measured(samples, [corner(1, 600.0)])
    assert m[1].turn_in == 560.0
    assert m[1].brake_at_turn_in == 100.0
    assert m[1].trail_m == pytest.approx(14.0)


def test_a_brake_released_before_turn_in_trails_nothing() -> None:
    samples = lap(brake=[(480.0, 540.0, 100.0)], yaw=[(560.0, 650.0, 0.6)])
    m = measured(samples, [corner(1, 600.0)])
    assert m[1].turn_in == 560.0
    assert m[1].brake_at_turn_in == 0.0
    assert m[1].trail_m == 0.0


def test_a_corner_that_runs_on_from_the_one_before_has_no_turn_in_of_its_own() -> None:
    samples = lap(yaw=[(300.0, 700.0, 0.5)])
    m = measured(samples, [corner(1, 450.0), corner(2, 600.0)])
    assert m[2].turn_in is None


def test_a_barely_turning_kink_has_no_turn_in() -> None:
    m = measured(lap(yaw=[(560.0, 650.0, 0.01)]), [corner(1, 600.0)])
    assert m[1].turn_in is None


def test_minimum_speed_and_where_it_was() -> None:
    samples = lap(brake=[(480.0, 580.0, 100.0)], slow=[(590.0, 620.0, 85.0)])
    m = measured(samples, [corner(1, 600.0)])
    assert m[1].min_speed == 85.0
    assert m[1].min_speed_dist == 590.0
    assert m[1].min_speed_gear == 3


def test_throttle_points_after_the_slowest_point() -> None:
    samples = lap(
        brake=[(480.0, 580.0, 100.0)],
        slow=[(590.0, 620.0, 85.0)],
        lift=[(580.0, 605.0)],
        part_throttle=[(605.0, 640.0, 50.0)],
    )
    m = measured(samples, [corner(1, 600.0)])
    assert m[1].flat is False
    assert m[1].throttle_on == 605.0
    assert m[1].throttle_full == 640.0


def test_a_corner_taken_flat_has_no_throttle_points() -> None:
    m = measured(lap(slow=[(590.0, 620.0, 170.0)]), [corner(1, 600.0)])
    assert m[1].flat is True
    assert m[1].throttle_on is None
    assert m[1].throttle_full is None
    assert m[1].brake_on is None


def test_full_throttle_is_not_looked_for_past_the_next_apex() -> None:
    samples = lap(
        brake=[(480.0, 580.0, 100.0)],
        slow=[(590.0, 620.0, 85.0)],
        lift=[(580.0, 600.0)],
        part_throttle=[(600.0, 900.0, 60.0)],
    )
    m = measured(samples, [corner(1, 600.0), corner(2, 700.0)])
    assert m[1].throttle_on == 600.0
    assert m[1].throttle_full is None


def test_a_corner_across_the_start_line_is_not_measured() -> None:
    wrapping = {"n": 9, "entry_dist": 1950.0, "apex_dist": 10.0, "exit_dist": 60.0}
    m = measured(lap(brake=[(1900.0, 1990.0, 100.0)]), [wrapping])
    assert m[9].brake_on is None
    assert m[9].min_speed is None


def test_a_corner_without_an_apex_takes_the_middle_of_its_window() -> None:
    (window,) = windows([{"n": 4, "entry_dist": 500.0, "exit_dist": 620.0}])
    assert window.apex == 560.0


def test_repeated_distances_keep_their_first_sample() -> None:
    """An aligned lap holds its place through a spin: the axis stands still
    while the clock runs. Distance stays a usable axis."""
    samples = {
        "dist": [0.0, 1.0, 1.0, 1.0, 2.0, 3.0],
        "t": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        "speed": [100.0, 90.0, 10.0, 20.0, 80.0, 100.0],
    }
    trace = LapTrace(samples)
    assert trace.dist == [0.0, 1.0, 2.0, 3.0]
    assert trace.column("speed") == [100.0, 90.0, 80.0, 100.0]
    assert trace.at("speed", 1.5) == pytest.approx(85.0)
    assert trace.at("speed", 3.5) is None
    assert trace.column("brake") is None


def test_deltas_need_both_laps_to_have_done_the_thing() -> None:
    corners = [corner(1, 600.0)]
    braked = measured(lap(brake=[(480.0, 580.0, 100.0)], slow=[(590.0, 620.0, 85.0)]), corners)
    earlier = measured(lap(brake=[(450.0, 580.0, 100.0)], slow=[(590.0, 620.0, 80.0)]), corners)
    flat = measured(lap(), corners)
    assert brake_point_delta(earlier[1], braked[1]) == -30.0
    assert min_speed_delta(earlier[1], braked[1]) == -5.0
    assert brake_point_delta(flat[1], braked[1]) is None
    assert brake_point_delta(braked[1], None) is None
    assert min_speed_delta(None, braked[1]) is None
