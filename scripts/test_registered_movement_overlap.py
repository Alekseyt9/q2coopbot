"""Physics telemetry attribution checks; no model inference or fitting."""
import unittest
from audit_registered_movement_overlap import classify


def row(velocity, ground, proposed=.5, applied=0, reason='unsupported_motion_guard'):
    return dict(observation=dict(velocity=velocity, on_ground=ground, enemies=[dict(clear_shot=True)]),
        proposed=dict(forward=proposed, side=0, vertical='release'),
        applied=dict(forward=applied, side=0, vertical='release'),
        selection=dict(interventions=[dict(component='movement', reason=reason)]))


class Tests(unittest.TestCase):
    def test_airborne_blocked_acceleration_can_leave_body_moving(self):
        counts, reasons = classify(row([200, 0, 0], False))
        self.assertEqual(counts['moving_suppressed_frames'], 1)
        self.assertEqual(counts['standing_clear_suppressed_frames'], 0)
        self.assertEqual(counts['unsupported_airborne_frames'], 1)
        self.assertFalse(reasons)

    def test_grounded_vertical_velocity_is_separate(self):
        counts, reasons = classify(row([0, 0, 20], True))
        self.assertEqual(counts['unsupported_grounded_vertical_velocity_frames'], 1)
        self.assertEqual(counts['standing_clear_suppressed_frames'], 1)
        self.assertEqual(reasons['unsupported_motion_guard'], 1)

    def test_no_intent_is_separate_from_blocked_intent(self):
        counts, _ = classify(row([0, 0, 0], True, proposed=0))
        self.assertEqual(counts['standing_clear_no_movement_intent_frames'], 1)
        self.assertEqual(counts['suppressed_frames'], 0)

    def test_applied_intent_does_not_prove_body_moves(self):
        counts, _ = classify(row([0, 0, 0], True, applied=.5))
        self.assertEqual(counts['standing_clear_applied_movement_frames'], 1)
        self.assertEqual(counts['standing_clear_frames'], 1)

    def test_unknown_unsupported_cause_and_invalid_velocity(self):
        counts, _ = classify(row([0, 0, 0], True))
        self.assertEqual(counts['unsupported_grounded_other_frames'], 1)
        with self.assertRaises(AssertionError):
            classify(row([float('nan'), 0, 0], True))


if __name__ == '__main__': unittest.main()
