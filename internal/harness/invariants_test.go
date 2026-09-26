package harness

import (
	"q2coopbot/internal/quake"
	"testing"
)

func lifecycleRows() (Scenario, []Trace, []Trace) {
	s := fixture()
	s.Expect.Invariants = []string{"wait_has_no_movement", "completed_search_stays_finished", "visible_contact_clears_search"}
	actor, bot := []Trace{}, []Trace{}
	for f := 40; f <= 46; f++ {
		actor = append(actor, Trace{Map: s.Map, Generation: 2, Frame: f, Scenario: &Status{State: "running", StepID: "test-step"}})
		bot = append(bot, Trace{Map: s.Map, Generation: 2, Frame: f, Goal: "wait_for_teammate"})
	}
	actor[6].Scenario = &Status{State: "completed", EndFrame: 46, CompletedSteps: 1}
	bot[0].Teammate = &quake.Vec3{}
	bot[1].Goal = "probe_last_seen"
	bot[1].SearchAttempt = &SearchAttempt{Entity: 2, LastSeenFrame: 40, State: "active"}
	bot[2].SearchAttempt = &SearchAttempt{Entity: 2, LastSeenFrame: 40, State: "completed", EndFrame: 42}
	return s, actor, bot
}
func TestLifecycleFirstFailure(t *testing.T) {
	for _, tc := range []struct {
		name   string
		mutate func([]Trace)
		frame  int
	}{
		{"wait_has_no_movement", func(b []Trace) { b[3].Command.Up = 1; b[4].Command.Forward = 100 }, 43},
		{"completed_search_stays_finished", func(b []Trace) { b[4].Goal = "probe_last_seen" }, 44},
		{"completed_search_stays_finished", func(b []Trace) { b[4].SearchAttempt = &SearchAttempt{Entity: 2, LastSeenFrame: 40, State: "active"} }, 44},
		{"visible_contact_clears_search", func(b []Trace) { b[1].Teammate = &quake.Vec3{} }, 41},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s, a, b := lifecycleRows()
			tc.mutate(b)
			r := Analyze(s, a, b)
			if r.State != "behavior_failed" || r.Reason != tc.name || r.Frame != tc.frame || len(r.Context) < 3 {
				t.Fatalf("%+v", r)
			}
			if r.Context[0].ActorStep != "test-step" {
				t.Fatal(r.Context)
			}
		})
	}
}
func TestLifecycleFreshObservationAndCoverage(t *testing.T) {
	s, a, b := lifecycleRows()
	age := 1
	b[4].Goal = "search_last_seen"
	b[4].TeammateAgeFrames = &age
	b[5].Goal = "probe_last_seen"
	b[5].SearchAttempt = &SearchAttempt{Entity: 2, LastSeenFrame: 43, State: "active"}
	if r := Analyze(s, a, b); r.State != "passed" {
		t.Fatalf("fresh observation rejected: %+v", r)
	}
	s, a, b = lifecycleRows()
	for i := range b {
		b[i].SearchAttempt = nil
	}
	if r := Analyze(s, a, b); r.State != "behavior_failed" || r.Reason != "invariant_not_exercised: completed_search_stays_finished" {
		t.Fatalf("%+v", r)
	}
	s.Expect.Invariants = []string{"typo"}
	if s.Validate() == nil {
		t.Fatal("unknown check accepted")
	}
}
