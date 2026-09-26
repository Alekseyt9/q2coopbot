package harness

import (
	"q2coopbot/internal/quake"
	"testing"
)

func failedEpisode() (Scenario, []Trace, []Trace) {
	s := fixture()
	target := quake.Vec3{100, 0, 0}
	s.Steps = []Step{{ID: "wait", Action: "walk", Timeout: 2, Target: &target}}
	s.Expect.Failure = &ExpectedFailure{Reason: "step_timeout", StepID: "wait", MinFrame: 42, MaxFrame: 42}
	var actor, bot []Trace
	for f := 40; f <= 42; f++ {
		actor = append(actor, Trace{Map: s.Map, Frame: f, Generation: 1})
		bot = append(bot, Trace{Map: s.Map, Frame: f, Generation: 1})
	}
	actor[2].Scenario = &Status{State: "failed", StepID: "wait", Reason: "step_timeout", EndFrame: 42}
	return s, actor, bot
}
func TestExpectedFailureRequiresExactEvidence(t *testing.T) {
	s, a, b := failedEpisode()
	r := Analyze(s, a, b)
	if !r.Accepted || r.State != "fixture_failed" || r.Expectation != "expected_failure_matched" {
		t.Fatalf("%+v", r)
	}
	for _, tc := range []struct {
		name   string
		change func(*Scenario, []Trace, []Trace)
	}{
		{"reason", func(s *Scenario, a, b []Trace) { a[2].Scenario.Reason = "actor_dead" }},
		{"step", func(s *Scenario, a, b []Trace) { a[2].Scenario.StepID = "other" }},
		{"frame", func(s *Scenario, a, b []Trace) { s.Expect.Failure.MinFrame = 43; s.Expect.Failure.MaxFrame = 43 }},
		{"success", func(s *Scenario, a, b []Trace) { a[2].Scenario.State = "completed"; a[2].Scenario.CompletedSteps = 1 }},
		{"generation", func(s *Scenario, a, b []Trace) { b[1].Generation = 2 }},
		{"gap", func(s *Scenario, a, b []Trace) { b[1].Frame = 40 }},
		{"invariant", func(s *Scenario, a, b []Trace) {
			s.Expect.Invariants = []string{"wait_has_no_movement"}
			b[0].Goal = "wait_for_teammate"
			b[0].Command.Forward = 10
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s, a, b := failedEpisode()
			tc.change(&s, a, b)
			r := Analyze(s, a, b)
			if r.Accepted || r.Expectation != "expected_failure_mismatch" {
				t.Fatalf("%+v", r)
			}
		})
	}
}
func TestExpectedFailureValidation(t *testing.T) {
	s, _, _ := failedEpisode()
	if err := s.Validate(); err != nil {
		t.Fatal(err)
	}
	for _, change := range []func(*ExpectedFailure){
		func(f *ExpectedFailure) { f.Reason = "any" }, func(f *ExpectedFailure) { f.StepID = "typo" }, func(f *ExpectedFailure) { f.MinFrame = 0 }, func(f *ExpectedFailure) { f.MaxFrame = 41 },
	} {
		s, _, _ := failedEpisode()
		change(s.Expect.Failure)
		if s.Validate() == nil {
			t.Fatal("accepted invalid expectation")
		}
	}
}
