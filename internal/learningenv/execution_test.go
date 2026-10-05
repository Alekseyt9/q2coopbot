package learningenv

import (
	"q2coopbot/internal/harness"
	"q2coopbot/internal/quake"
	"testing"
)

func TestExecutionRequiresExactIdentityCommandAndDispatchWindow(t *testing.T) {
	s := damageStep(10)
	s.ClientSequence = 12
	s.Command = quake.UserCmd{Forward: 100, Msec: 100}
	a := harness.AppliedCommand{Connection: 1, Generation: 2, Sequence: 12, Frame: 10, Kind: "new", Command: s.Command}
	e := NewExecutionIndex([]harness.AppliedCommand{a}).Match(s)
	if !e.Matched || !e.WindowExclusive || e.DispatchFrame == nil || *e.DispatchFrame != 10 {
		t.Fatal(e)
	}
	for _, change := range []func(*harness.AppliedCommand){
		func(c *harness.AppliedCommand) { c.Connection++ }, func(c *harness.AppliedCommand) { c.Generation++ },
		func(c *harness.AppliedCommand) { c.Sequence++ }, func(c *harness.AppliedCommand) { c.Kind = "old" },
		func(c *harness.AppliedCommand) { c.Command.Forward++ },
	} {
		b := a
		change(&b)
		e = NewExecutionIndex([]harness.AppliedCommand{b}).Match(s)
		if e.Matched || e.Reason == "" {
			t.Fatal(e)
		}
	}
	for _, frame := range []int{9, 11} {
		b := a
		b.Frame = frame
		e = NewExecutionIndex([]harness.AppliedCommand{b}).Match(s)
		if !e.Matched || e.WindowExclusive || e.Reason != "dispatch_outside_observation_window" {
			t.Fatal(e)
		}
	}
	e = NewExecutionIndex([]harness.AppliedCommand{a, a}).Match(s)
	if e.Matched {
		t.Fatal("duplicate dispatch accepted")
	}
	b := a
	b.Kind = "old"
	b.Command.Forward = 0
	e = NewExecutionIndex([]harness.AppliedCommand{b, a}).Match(s)
	if !e.Matched || e.WindowExclusive || e.RecoveryCommands != 1 {
		t.Fatal("recovery disguised as exclusive execution", e)
	}
	s.Next = nil
	e = NewExecutionIndex([]harness.AppliedCommand{a}).Match(s)
	if !e.Matched || e.WindowExclusive {
		t.Fatal("tail fabricated next-frame evidence", e)
	}
}
