package harness

import (
	"q2coopbot/internal/quake"
	"testing"
)

func TestMotionUsesPreviousCommandAndExcludesPlacement(t *testing.T) {
	s := fixture()
	s.Steps = append(s.Steps, Step{ID: "place", Action: "place"})
	rows := []Trace{
		{Self: &quake.Vec3{0, 0, 0}, Command: quake.UserCmd{Forward: 100}},
		{Self: &quake.Vec3{3, 4, 0}, Command: quake.UserCmd{Forward: 100}},
		{Self: &quake.Vec3{3, 4, 0}, Command: quake.UserCmd{Forward: 100}, Scenario: &Status{StepID: "place"}},
		{Self: &quake.Vec3{1003, 4, 0}, Scenario: &Status{StepID: "place"}},
	}
	m := measureMotion(s, rows, true)
	if m.PathUnits != 5 || m.CommandedIntervals != 2 || m.LowDisplacementIntervals != 1 || m.LongestLowDisplacementRun != 1 || m.ExcludedPlacementIntervals != 1 {
		t.Fatalf("%+v", m)
	}
	rows[1].Self = nil
	if measureMotion(s, rows, true) != nil {
		t.Fatal("missing observation reported as zero movement")
	}
}

func TestFailedWalkRetainsProgress(t *testing.T) {
	s := fixture()
	target := quake.Vec3{100, 0, 0}
	s.Steps = []Step{{ID: "walk", Action: "walk", Target: &target, Timeout: 2}}
	var rows []Trace
	for f := 40; f <= 42; f++ {
		rows = append(rows, Trace{Map: s.Map, Frame: f, Generation: 1, Self: &quake.Vec3{float64(f-40) * 10, 0, 0}, Scenario: &Status{State: "running", StepID: "walk"}})
	}
	rows[2].Scenario.State = "failed"
	rows[2].Scenario.Reason = "step_timeout"
	r := Analyze(s, rows, rows)
	if r.State != "fixture_failed" || r.Metrics.ActorMotion.PathUnits != 20 || len(r.Metrics.WalkSteps) != 1 || r.Metrics.WalkSteps[0].Progress != 20 || r.Metrics.WalkSteps[0].Completed || len(r.Context) != 3 {
		t.Fatalf("%+v", r)
	}
}
