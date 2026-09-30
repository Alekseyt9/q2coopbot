package harness

import "testing"

func TestWaitSignalRequiresMatchingReleaseAndTimesOut(t *testing.T) {
	s := fixture()
	s.GameFrames = 200
	s.Steps = []Step{{ID: "release-player", Action: "wait_signal", Timeout: 3}}
	if err := s.Validate(); err != nil {
		t.Fatal(err)
	}
	r := New(s)
	for f := 40; f <= 41; f++ {
		in := input(f)
		in.StepSignal = "wrong-step"
		r.Tick(in)
		if r.Status.State != "running" {
			t.Fatal("released by wrong signal", r.Status)
		}
	}
	in := input(42)
	in.StepSignal = "release-player"
	r.Tick(in)
	if r.Status.State != "completed" {
		t.Fatal("matching signal ignored", r.Status)
	}
	r = New(s)
	for f := 40; f <= 43; f++ {
		r.Tick(input(f))
	}
	if r.Status.Reason != "step_timeout" {
		t.Fatal("missing signal did not time out", r.Status)
	}
	s.Steps[0].ID = "../escape"
	if s.Validate() == nil {
		t.Fatal("unsafe signal filename accepted")
	}
}
