"""
Unit tests for PAT finite state machine transitions and timing.
"""

import pytest
from tracking.state_machine import TrackingStateMachine, PATState


def test_state_machine_nominal_acquisition():
    fsm = TrackingStateMachine(acquire_confirm_frames=3)
    assert fsm.current_state == PATState.SEARCHING

    # Frame 1: Detection spotted -> ACQUIRING
    state, changed = fsm.update(has_detection=True, sim_time=0.1)
    assert state == PATState.ACQUIRING
    assert changed is True

    # Frame 2: Still acquiring (needs 3 consecutive frames)
    state, changed = fsm.update(has_detection=True, sim_time=0.2)
    assert state == PATState.ACQUIRING
    assert changed is False

    # Frame 3: Confirmed -> TRACKING
    state, changed = fsm.update(has_detection=True, sim_time=0.3)
    assert state == PATState.TRACKING
    assert changed is True


def test_state_machine_temporary_loss_recovery():
    fsm = TrackingStateMachine(acquire_confirm_frames=2, uncertain_timeout_frames=5)
    fsm.reset(initial_state=PATState.TRACKING)

    # 1 miss -> UNCERTAIN (Kalman coasting)
    state, changed = fsm.update(has_detection=False, sim_time=1.0)
    assert state == PATState.UNCERTAIN
    assert changed is True

    # 2 more misses -> Still UNCERTAIN
    fsm.update(has_detection=False, sim_time=1.1)
    state, _ = fsm.update(has_detection=False, sim_time=1.2)
    assert state == PATState.UNCERTAIN

    # Detection reappears -> Immediate recovery to TRACKING
    state, changed = fsm.update(has_detection=True, sim_time=1.3)
    assert state == PATState.TRACKING
    assert changed is True


def test_state_machine_timeout_to_lost_and_reacquire():
    fsm = TrackingStateMachine(
        acquire_confirm_frames=2,
        uncertain_timeout_frames=3,
        reacquire_max_frames=10,
    )
    fsm.reset(initial_state=PATState.TRACKING)

    # 3 consecutive misses (at uncertain_timeout=3)
    for i in range(3):
        fsm.update(has_detection=False, sim_time=1.0 + i * 0.1)

    # Miss 4 -> Transition to LOST (consecutive_misses=4 > uncertain_timeout)
    state, changed = fsm.update(has_detection=False, sim_time=1.4)
    assert state == PATState.LOST
    assert fsm.total_lost_events >= 1

    # Miss 5 -> Transition to REACQUIRING
    state, changed = fsm.update(has_detection=False, sim_time=1.5)
    assert state == PATState.REACQUIRING
