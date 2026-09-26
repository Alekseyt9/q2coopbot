package harness

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

func sessionFixture() Session {
	a, b := fixture(), fixture()
	b.Map = "base3"
	return Session{Version: 1, Name: "maps", TransitionTimeoutMS: 1000, Phases: []Phase{{ID: "first", Scenario: a}, {ID: "second", Scenario: b}}}
}

func TestSessionWaitsForDynamicBarrier(t *testing.T) {
	s := sessionFixture()
	s.ReadinessBarrier = true
	r, err := NewSession(s)
	if err != nil {
		t.Fatal(err)
	}
	in := input(80)
	if d := r.Tick(in, 0); r.Status.State != "pending" || d.ChangeMap != "" {
		t.Fatal(r.Status)
	}
	in.PhaseStart = 90
	r.Tick(in, time.Millisecond)
	if r.Status.StartFrame != 90 || r.Status.Phase.CompletedSteps != 0 {
		t.Fatal(r.Status)
	}
	for f := 90; f <= 92; f++ {
		in.Frame = f
		r.Tick(in, time.Duration(f)*time.Millisecond)
	}
	if r.Status.State != "waiting_map" || r.Status.Phase.EndFrame != 92 {
		t.Fatal(r.Status)
	}
	in.Map = "base3"
	in.Generation = 3
	in.Frame = 120
	in.PhaseStart = 0
	r.Tick(in, 100*time.Millisecond)
	if r.Status.StartFrame != 0 || r.Status.State != "pending" {
		t.Fatal(r.Status)
	}
	in.PhaseStart = 119
	r.Tick(in, 101*time.Millisecond)
	if r.Status.Reason != "barrier_start_missed" {
		t.Fatal(r.Status)
	}
}

func waitingSession(t *testing.T) *SessionRunner {
	t.Helper()
	r, err := NewSession(sessionFixture())
	if err != nil {
		t.Fatal(err)
	}
	for frame := 40; frame <= 42; frame++ {
		d := r.Tick(input(frame), time.Duration(frame)*time.Millisecond)
		if frame < 42 && d.ChangeMap != "" {
			t.Fatal("transition before phase completion")
		}
		if frame == 42 && (d.ChangeMap != "base3" || r.Status.State != "waiting_map") {
			t.Fatalf("%+v %+v", d, r.Status)
		}
	}
	return r
}

func TestSessionTransitionsAndCompletes(t *testing.T) {
	r := waitingSession(t)
	for n := 0; n < 3; n++ {
		d := r.Tick(input(42), 50*time.Millisecond)
		if d.ChangeMap != "" || d.Kill || d.Place != nil || d.Walk != nil {
			t.Fatal("repeated transition acted")
		}
	}
	in := input(1)
	in.Map = "base3"
	in.Generation = 1
	r.Tick(in, 60*time.Millisecond)
	if r.Status.PhaseIndex != 1 || r.Status.PhaseID != "second" || r.Status.State != "pending" {
		t.Fatal(r.Status)
	}
	for f := 40; f <= 42; f++ {
		in.Frame = f
		r.Tick(in, time.Duration(100+f)*time.Millisecond)
	}
	if r.Status.State != "completed" || r.Status.Phase.CompletedSteps != 1 || r.Status.Location.Generation != 1 {
		t.Fatal(r.Status)
	}
	if d := r.Tick(input(40), time.Second); d.ChangeMap != "" || r.Status.State != "completed" {
		t.Fatal("terminal state changed")
	}
}

func TestSessionTransitionFailures(t *testing.T) {
	for _, tc := range []struct {
		name    string
		in      Input
		elapsed time.Duration
		reason  string
	}{
		{"silent", input(42), 1042 * time.Millisecond, "map_transition_timeout"},
		{"deadline arrival", Input{Map: "base3", Generation: 3, Frame: 1, Health: 100}, 1042 * time.Millisecond, "map_transition_timeout"},
		{"wrong map", Input{Map: "base1", Generation: 3, Frame: 1, Health: 100}, 50 * time.Millisecond, "unexpected_transition_map"},
		{"same generation", Input{Map: "base3", Generation: 2, Frame: 1, Health: 100}, 50 * time.Millisecond, "map_changed_without_generation"},
		{"clock", input(42), 41 * time.Millisecond, "clock_reversed"},
		{"late", Input{Map: "base3", Generation: 3, Frame: 41, Health: 100}, 50 * time.Millisecond, "late_start"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			r := waitingSession(t)
			r.Tick(tc.in, tc.elapsed)
			if r.Status.State != "failed" || r.Status.Reason != tc.reason {
				t.Fatal(r.Status)
			}
			if d := r.Tick(input(43), 2*time.Second); d.ChangeMap != "" || r.Status.Reason != tc.reason {
				t.Fatal("failure changed")
			}
		})
	}
}

func TestSessionReloadAndGenerationDuringSetup(t *testing.T) {
	s := sessionFixture()
	s.Phases[1].Scenario.Map = s.Phases[0].Scenario.Map
	r, err := NewSession(s)
	if err != nil {
		t.Fatal(err)
	}
	for f := 40; f <= 42; f++ {
		r.Tick(input(f), time.Duration(f)*time.Millisecond)
	}
	r.Tick(input(1), 50*time.Millisecond)
	if r.Status.State != "waiting_map" {
		t.Fatal("same map mistaken for reload")
	}
	in := input(1)
	in.Generation = 7
	r.Tick(in, 60*time.Millisecond)
	if r.Status.PhaseIndex != 1 {
		t.Fatal(r.Status)
	}
	in.Generation = 8
	in.Frame = 2
	r.Tick(in, 70*time.Millisecond)
	if r.Status.Reason != "unexpected_phase_generation" {
		t.Fatal(r.Status)
	}
}

func TestSessionPhaseFailureNeverTransitions(t *testing.T) {
	r, _ := NewSession(sessionFixture())
	r.Tick(input(40), 0)
	in := input(41)
	in.Health = 0
	if d := r.Tick(in, time.Millisecond); d.ChangeMap != "" || r.Status.Reason != "actor_dead" {
		t.Fatal(r.Status)
	}
}

func TestSessionRejectsRevisitedGeneration(t *testing.T) {
	s := sessionFixture()
	third := fixture()
	s.Phases = append(s.Phases, Phase{ID: "third", Scenario: third})
	r, err := NewSession(s)
	if err != nil {
		t.Fatal(err)
	}
	for f := 40; f <= 42; f++ {
		r.Tick(input(f), time.Duration(f)*time.Millisecond)
	}
	in := input(40)
	in.Map, in.Generation = "base3", 3
	for f := 40; f <= 42; f++ {
		in.Frame = f
		r.Tick(in, time.Duration(f+10)*time.Millisecond)
	}
	if r.Status.State != "waiting_map" {
		t.Fatal(r.Status)
	}
	r.Tick(input(1), 70*time.Millisecond)
	if r.Status.Reason != "generation_revisited" {
		t.Fatal(r.Status)
	}
}

func TestSessionPropagatesRouteFailure(t *testing.T) {
	s := sessionFixture()
	s.Phases[0].Scenario.Steps = []Step{{ID: "route", Action: "walk", Target: &s.Phases[0].Scenario.ActorOrigin, Timeout: 2, Route: true}}
	r, err := NewSession(s)
	if err != nil {
		t.Fatal(err)
	}
	r.Tick(input(40), 0)
	r.RejectRoute("route_unavailable")
	if r.Status.State != "failed" || r.Status.Phase.StepID != "route" || r.Status.Reason != "route_unavailable" {
		t.Fatal(r.Status)
	}
	if d := r.Tick(input(41), time.Millisecond); d.ChangeMap != "" {
		t.Fatal("transition after route failure")
	}
}

func TestSessionStrictDefinition(t *testing.T) {
	for _, mutate := range []func(*Session){
		func(s *Session) { s.Phases = s.Phases[:1] },
		func(s *Session) { s.Phases[1].ID = s.Phases[0].ID },
		func(s *Session) { s.TransitionTimeoutMS = 0 },
		func(s *Session) { s.Phases[1].Scenario.Map = "base3;quit" },
		func(s *Session) { s.Phases[1].Scenario.Expect.MapSequence = []string{"base2", "base3"} },
	} {
		s := sessionFixture()
		mutate(&s)
		if _, err := NewSession(s); err == nil {
			t.Fatal("invalid session accepted")
		}
	}
	path := filepath.Join(t.TempDir(), "session.json")
	if err := os.WriteFile(path, []byte(`{"version":1,"unknown":true}`), 0600); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadSession(path); err == nil {
		t.Fatal("unknown field accepted")
	}
}
