package harness

import "testing"

func TestRunnerCheckpointResumesRemainingBudgetWithoutReplayingEntry(t *testing.T) {
	s := Scenario{Version: 1, Name: "checkpoint-wait", Map: "base1", StartFrame: 40, GameFrames: 100, Steps: []Step{{ID: "wait", Action: "wait", Frames: 20}}}
	r := New(s)
	for frame := 40; frame <= 45; frame++ {
		r.Tick(Input{Map: "base1", Frame: frame, Generation: 1, Health: 100, OnGround: true})
	}
	state, err := r.CaptureCheckpoint()
	if err != nil {
		t.Fatal(err)
	}
	fresh := Input{Map: "base1", Frame: 2, Generation: 4, Health: 100, OnGround: true}
	a, err := RestoreRunnerCheckpoint(s, state, fresh)
	if err != nil {
		t.Fatal(err)
	}
	b, err := RestoreRunnerCheckpoint(s, state, fresh)
	if err != nil {
		t.Fatal(err)
	}
	for frame := 2; frame <= 17; frame++ {
		fresh.Frame = frame
		d := a.Tick(fresh)
		if d.NewStep || d.Kill || d.Place != nil {
			t.Fatal("entry replayed")
		}
	}
	if a.Status.State != "completed" || a.Status.EndFrame != 17 || b.Status.State != "running" {
		t.Fatal(a.Status, b.Status)
	}
	changed := s
	changed.Steps = append([]Step(nil), s.Steps...)
	changed.Steps[0].Frames++
	if _, err = RestoreRunnerCheckpoint(changed, state, fresh); err == nil {
		t.Fatal("changed scenario accepted")
	}
	state.Elapsed = -1
	if _, err = RestoreRunnerCheckpoint(s, state, fresh); err == nil {
		t.Fatal("future progress accepted")
	}
}
