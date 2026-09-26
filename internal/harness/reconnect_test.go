package harness

import (
	"testing"
	"time"
)

func TestReconnectRequiresNewConnectionOnSameGeneration(t *testing.T) {
	s := sessionFixture()
	s.Phases[1].Scenario.Map = s.Phases[0].Scenario.Map
	s.Phases[1].Entry = "reconnect"
	s.ReadinessBarrier = true
	r, err := NewSession(s)
	if err != nil {
		t.Fatal(err)
	}
	in := input(40)
	in.Connection = 1
	in.PhaseStart = 40
	for f := 40; f <= 42; f++ {
		in.Frame = f
		r.Tick(in, time.Duration(f)*time.Millisecond)
	}
	in.Frame = 50
	in.PhaseStart = 0
	r.Tick(in, 50*time.Millisecond)
	if r.Status.State != "waiting_map" {
		t.Fatal("old connection accepted", r.Status)
	}
	in.Connection = 2
	r.Tick(in, 51*time.Millisecond)
	if r.Status.PhaseIndex != 1 || r.Status.State != "pending" {
		t.Fatal(r.Status)
	}
	in.PhaseStart = 60
	for f := 60; f <= 62; f++ {
		in.Frame = f
		r.Tick(in, time.Duration(f)*time.Millisecond)
	}
	if r.Status.State != "completed" || r.Status.Location.Connection != 2 {
		t.Fatal(r.Status)
	}
}

func TestReconnectReportSeparatesEqualGeneration(t *testing.T) {
	s, a, b := sessionReportTraces()
	s.Phases[1].Entry = "reconnect"
	s.Phases[1].Scenario.Map = s.Phases[0].Scenario.Map
	s.Phases[1].Scenario.StartFrame = 60
	for i := range a {
		a[i].Connection = 1
		b[i].Connection = 1
		if i >= 3 {
			a[i].Map = a[0].Map
			b[i].Map = b[0].Map
			a[i].Generation = a[0].Generation
			b[i].Generation = b[0].Generation
			a[i].Connection = 2
			b[i].Connection = 2
			a[i].Frame += 20
			b[i].Frame += 20
			a[i].Scenario.StepStart += 20
			if a[i].Scenario.EndFrame > 0 {
				a[i].Scenario.EndFrame += 20
			}
		}
	}
	if r := AnalyzeSession(s, a, b); !r.Accepted || r.Phases[1].Connection != 2 {
		t.Fatalf("%+v", r)
	}
	s.Phases[1].Entry = ""
	if r := AnalyzeSession(s, a, b); r.Accepted || r.Reason != "new map generation not observed" {
		t.Fatalf("%+v", r)
	}
}

func TestAppliedCommandsSeparateConnections(t *testing.T) {
	sent := []Trace{{Connection: 1, Generation: 1, ClientSequence: 10}, {Connection: 2, Generation: 1, ClientSequence: 10}}
	applied := []AppliedCommand{{Connection: 1, Generation: 1, Sequence: 10, Kind: "new"}, {Connection: 2, Generation: 1, Sequence: 10, Kind: "new"}}
	if r := VerifyAppliedCommands(sent, applied); !r.Accepted || !r.ConnectionIdentity {
		t.Fatal(r)
	}
	applied[1].Connection = 1
	if r := VerifyAppliedCommands(sent, applied); r.Accepted {
		t.Fatal("aliased connections accepted")
	}
}
