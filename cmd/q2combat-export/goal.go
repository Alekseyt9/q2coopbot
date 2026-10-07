package main

import (
	"fmt"

	"q2coopbot/internal/learningenv"
	"q2coopbot/internal/policy"
)

func markGoalBoundary(s *learningenv.Step) error {
	// A final learned command can kill the target and hand control back to
	// rules on the next observation. Preserve that segment boundary and its
	// verified kill reward; do not reinterpret it as a regular policy step.
	if s.Next == nil || s.Terminal || s.Next.Health <= 0 || s.Next.Identity.Life != 1 ||
		!policy.SameLife(s.Observation.Identity, s.Next.Identity) ||
		s.Next.Identity.Frame != s.Observation.Identity.Frame+1 ||
		(s.Truncated && s.Reason != "control_handoff") {
		return fmt.Errorf("goal boundary lacks a complete living first-life transition")
	}
	if !s.Truncated {
		s.Terminal, s.Reason = true, "combat_goal_complete"
	}
	return nil
}
