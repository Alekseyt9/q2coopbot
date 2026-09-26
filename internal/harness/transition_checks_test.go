package harness

import (
	"q2coopbot/internal/quake"
	"testing"
)

func TestReconnectSearchCheckCannotBeSatisfiedByMapChange(t *testing.T) {
	rows := []Trace{{Map: "base2", Generation: 7, Connection: 1, Frame: 50, Goal: "probe_last_seen"}, {Map: "base2", Generation: 7, Connection: 2, Frame: 80, Goal: "wait_for_teammate"}}
	checks := checkSearchBoundary(rows, "search_reset_on_reconnect")
	if len(checks) != 1 || !checks[0].Passed || checks[0].After.Connection != 2 {
		t.Fatal(checks)
	}
	rows[1].SearchAttempt = &SearchAttempt{State: "completed"}
	if checkSearchBoundary(rows, "search_reset_on_reconnect")[0].Passed {
		t.Fatal("old attempt survived reconnect")
	}
	rows[1].Generation = 8
	if len(checkSearchBoundary(rows, "search_reset_on_reconnect")) != 0 {
		t.Fatal("map transition substituted for reconnect")
	}
	rows[1].Generation = 7
	rows[0].Goal = "wait_for_teammate"
	if len(checkSearchBoundary(rows, "search_reset_on_reconnect")) != 0 {
		t.Fatal("unexercised search accepted")
	}
}

func TestSearchTransitionRequiresExerciseAndClearNewDecision(t *testing.T) {
	s, a, b := sessionReportTraces()
	s.Invariants = []string{"search_reset_on_transition"}
	if r := AnalyzeSession(s, a, b); r.Accepted || r.Reason != "invariant_not_exercised: search_reset_on_transition" {
		t.Fatal(r)
	}
	b[2].Goal = "search_last_seen"
	if r := AnalyzeSession(s, a, b); !r.Accepted || len(r.TransitionChecks) != 1 || r.TransitionChecks[0].Before.Generation == r.TransitionChecks[0].After.Generation {
		t.Fatal(r)
	}
	for _, mutate := range []func(*Trace){func(r *Trace) { r.Goal = "probe_last_seen" }, func(r *Trace) { r.SearchTarget = &quake.Vec3{} }, func(r *Trace) { r.SearchAttempt = &SearchAttempt{State: "completed"} }} {
		bad := append([]Trace(nil), b...)
		mutate(&bad[3])
		if r := AnalyzeSession(s, a, bad); r.Accepted || r.Reason != "search_state_after_transition" {
			t.Fatal(r)
		}
	}
	// New search later in the new generation is not itself stale state.
	b[4].Goal = "search_last_seen"
	if r := AnalyzeSession(s, a, b); !r.Accepted {
		t.Fatal(r)
	}
}
