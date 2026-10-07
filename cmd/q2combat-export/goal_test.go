package main

import (
	"testing"

	"q2coopbot/internal/learningenv"
	"q2coopbot/internal/policy"
)

func TestGoalBoundaryPreservesCompleteHandoff(t *testing.T) {
	id := policy.Identity{Life: 1, Frame: 174}
	next := id
	next.Frame++
	s := learningenv.Step{Observation: policy.Observation{Identity: id, Health: 80}, Next: &policy.Observation{Identity: next, Health: 80}, Truncated: true, Reason: "control_handoff"}
	if err := markGoalBoundary(&s); err != nil || !s.Truncated || s.Terminal || s.Reason != "control_handoff" {
		t.Fatalf("handoff changed or rejected: %+v %v", s, err)
	}
	s.Reason = "frame_gap"
	if err := markGoalBoundary(&s); err == nil {
		t.Fatal("invalid transition accepted as goal")
	}
	s.Truncated, s.Reason = false, ""
	if err := markGoalBoundary(&s); err != nil || !s.Terminal || s.Reason != "combat_goal_complete" {
		t.Fatalf("regular goal not marked: %+v %v", s, err)
	}
}
