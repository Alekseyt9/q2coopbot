package harness

import (
	"strings"
	"testing"
)

func mapSequenceRows() []Trace {
	return []Trace{
		{Map: "base1", Generation: 91, Frame: 40},
		{Map: "base1", Generation: 91, Frame: 41},
		{Map: "base2", Generation: 17, Frame: 40},
		{Map: "base2", Generation: 17, Frame: 41},
		{Map: "base2", Generation: 17, Frame: 42, Scenario: &Status{State: "completed", CompletedSteps: 1, EndFrame: 42}},
	}
}

func TestMapSequenceAcceptance(t *testing.T) {
	s := fixture()
	s.Expect.MapSequence = []string{"base1", "base2"}
	if err := s.Validate(); err != nil {
		t.Fatal(err)
	}
	r := Analyze(s, mapSequenceRows(), mapSequenceRows())
	if !r.Accepted {
		t.Fatalf("%+v", r)
	}
	for _, tc := range []struct {
		name   string
		mutate func([]Trace) []Trace
		reason string
	}{
		{"missing", func(rows []Trace) []Trace { return rows[2:] }, "expected 2 segments"},
		{"wrong map", func(rows []Trace) []Trace { rows[0].Map = "base3"; rows[1].Map = "base3"; return rows }, "expected base1"},
		{"wrong generation", func(rows []Trace) []Trace { rows[0].Generation = 90; rows[1].Generation = 90; return rows }, "generation mismatch"},
		{"disjoint", func(rows []Trace) []Trace { rows[0].Frame = 50; rows[1].Frame = 51; return rows }, "no shared frame"},
		{"gap", func(rows []Trace) []Trace { rows[1].Frame = 43; return rows }, "frame_gap"},
		{"extra", func(rows []Trace) []Trace { return append(rows, Trace{Map: "base3", Generation: 18, Frame: 1}) }, "expected 2 segments"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			r := Analyze(s, mapSequenceRows(), tc.mutate(mapSequenceRows()))
			if r.Accepted || r.State != "trace_invalid" || !strings.Contains(r.Reason, tc.reason) {
				t.Fatalf("%+v", r)
			}
		})
	}
}

func TestMapSequenceReloadAndOpaqueGeneration(t *testing.T) {
	rows := mapSequenceRows()
	for i := 2; i < len(rows); i++ {
		rows[i].Map = "base1"
	}
	timeline := SessionTimeline{Actor: traceTimeline(rows), Bot: traceTimeline(rows)}
	if reason, _ := verifyMapSequence([]string{"base1", "base1"}, timeline); reason != "" {
		t.Fatal(reason)
	}
	for i := 2; i < len(rows); i++ {
		rows[i].Generation = 91
	}
	timeline = SessionTimeline{Actor: traceTimeline(rows), Bot: traceTimeline(rows)}
	if reason, _ := verifyMapSequence([]string{"base1", "base1"}, timeline); reason == "" {
		t.Fatal("reload without new generation accepted")
	}
}

func TestMapSequenceValidation(t *testing.T) {
	for _, sequence := range [][]string{{"base2"}, {"base1", "base3"}, {"bad;quit", "base2"}} {
		s := fixture()
		s.Expect.MapSequence = sequence
		if s.Validate() == nil {
			t.Fatalf("accepted %v", sequence)
		}
	}
}

func TestExpectedFailureCannotHideMissingMap(t *testing.T) {
	s := fixture()
	s.Steps = []Step{{ID: "wait", Action: "respawn_cycle", Timeout: 2}}
	s.Expect.MapSequence = []string{"base1", "base2"}
	s.Expect.Failure = &ExpectedFailure{Reason: "step_timeout", StepID: "wait", MinFrame: 42, MaxFrame: 42}
	if err := s.Validate(); err != nil {
		t.Fatal(err)
	}
	actor := mapSequenceRows()
	actor[4].Scenario = &Status{State: "failed", Reason: "step_timeout", StepID: "wait", EndFrame: 42}
	r := Analyze(s, actor, mapSequenceRows()[2:])
	if r.Accepted || r.State != "trace_invalid" {
		t.Fatalf("%+v", r)
	}
}
