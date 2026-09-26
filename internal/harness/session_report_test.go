package harness

import "testing"

func TestSessionReportChecksBarrierAgreement(t *testing.T) {
	s, a, b := sessionReportTraces()
	s.ReadinessBarrier = true
	for i := range a {
		a[i].Session.StartFrame = 40
		a[i].SessionStartFrame = 40
		b[i].SessionStartFrame = 40
	}
	if r := AnalyzeSession(s, a, b); !r.Accepted {
		t.Fatalf("%+v", r)
	}
	b[3].SessionStartFrame = 41
	if r := AnalyzeSession(s, a, b); r.Accepted {
		t.Fatal("different observer start accepted")
	}
	b[3].SessionStartFrame = 40
	before := b[0]
	before.Frame = 39
	before.Command.Forward = 100
	b = append([]Trace{before}, b...)
	if r := AnalyzeSession(s, a, b); r.Accepted || r.Reason != "command before session barrier" {
		t.Fatalf("%+v", r)
	}
}

func sessionReportTraces() (Session, []Trace, []Trace) {
	s := sessionFixture()
	var actor, bot []Trace
	for i, phase := range s.Phases {
		for frame := 40; frame <= 42; frame++ {
			status := Status{State: "running", StepID: "wait", StepStart: 40}
			state := "running"
			if frame == 42 {
				status.State = "completed"
				status.CompletedSteps = 1
				status.EndFrame = 42
				state = "waiting_map"
				if i == 1 {
					state = "completed"
				}
			}
			session := SessionStatus{State: state, PhaseID: phase.ID, PhaseIndex: i, Phase: status}
			actor = append(actor, Trace{Map: phase.Scenario.Map, Generation: 10 + i, Frame: frame, Scenario: &status, Session: &session})
			bot = append(bot, Trace{Map: phase.Scenario.Map, Generation: 10 + i, Frame: frame})
		}
	}
	return s, actor, bot
}

func TestSessionReportSeparatesLocalFrames(t *testing.T) {
	s, a, b := sessionReportTraces()
	r := AnalyzeSession(s, a, b)
	if !r.Accepted || len(r.Phases) != 2 || r.Phases[0].Report.Metrics.Frames != 3 || r.Phases[1].Report.Metrics.Frames != 3 || r.Phases[0].Generation == r.Phases[1].Generation {
		t.Fatalf("%+v", r)
	}
}

func TestSessionReportRejectsIncompleteAndCorruptTraces(t *testing.T) {
	for _, tc := range []struct {
		name   string
		mutate func([]Trace, []Trace) ([]Trace, []Trace)
	}{
		{"missing phase", func(a, b []Trace) ([]Trace, []Trace) { return a[:3], b[:3] }},
		{"missing frame", func(a, b []Trace) ([]Trace, []Trace) { return a, append(b[:4], b[5:]...) }},
		{"wrong identity", func(a, b []Trace) ([]Trace, []Trace) { a[3].Session.PhaseID = "first"; return a, b }},
		{"no final completion", func(a, b []Trace) ([]Trace, []Trace) { a[5].Session.State = "waiting_map"; return a, b }},
		{"phase incomplete", func(a, b []Trace) ([]Trace, []Trace) { a[2].Scenario.State = "running"; return a, b }},
		{"different generation", func(a, b []Trace) ([]Trace, []Trace) {
			for i := 3; i < len(b); i++ {
				b[i].Generation = 22
			}
			return a, b
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s, a, b := sessionReportTraces()
			a, b = tc.mutate(a, b)
			if r := AnalyzeSession(s, a, b); r.Accepted {
				t.Fatalf("%+v", r)
			}
		})
	}
}
