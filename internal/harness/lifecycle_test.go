package harness

import "testing"

func TestRespawnCycleRequiresDeathBeforeAliveAndResets(t *testing.T) {
	s := fixture()
	s.Steps = []Step{{ID: "one", Action: "respawn_cycle", Timeout: 10}, {ID: "two", Action: "respawn_cycle", Timeout: 10}}
	r := New(s)
	if !r.Tick(input(40)).Kill {
		t.Fatal("missing kill")
	}
	if d := r.Tick(input(40)); d.Kill {
		t.Fatal("repeated kill")
	}
	r.Tick(input(41))
	if r.Status.CompletedSteps != 0 {
		t.Fatal("alive without death accepted")
	}
	dead := input(42)
	dead.Health = 0
	if !r.Tick(dead).Respawn || r.Status.DeathFrame != 42 {
		t.Fatal(r.Status)
	}
	r.Tick(input(43))
	if r.Status.CompletedSteps != 1 || r.Status.RespawnFrame != 43 {
		t.Fatal(r.Status)
	}
	if !r.Tick(input(44)).Kill || r.Status.DeathFrame != 0 {
		t.Fatal("old death retained")
	}
	r.Tick(input(45))
	if r.Status.CompletedSteps != 1 {
		t.Fatal("old death completed second cycle")
	}
	dead.Frame = 46
	r.Tick(dead)
	r.Tick(input(47))
	if r.Status.State != "completed" {
		t.Fatal(r.Status)
	}
}
func TestLifecycleMissingHealthDoesNotInventTransitions(t *testing.T) {
	alive, dead := int16(100), int16(0)
	r := []Trace{{Frame: 40, Health: &alive}, {Frame: 41, Health: &dead}, {Frame: 42, Health: &alive}}
	e := actorLifecycle(r)
	if len(e) != 2 || e[0].Kind != "actor_died" || e[1].Kind != "actor_respawned" {
		t.Fatal(e)
	}
	r[1].Health = nil
	if len(actorLifecycle(r)) != 0 {
		t.Fatal("missing health interpreted as death")
	}
}

func TestCompletedRespawnClaimNeedsObservedHealthCycle(t *testing.T) {
	s := fixture()
	s.Steps = []Step{{ID: "cycle", Action: "respawn_cycle", Timeout: 10}}
	alive, dead := int16(100), int16(0)
	rows := []Trace{}
	for f := 40; f <= 42; f++ {
		rows = append(rows, Trace{Map: s.Map, Frame: f, Generation: 1, Health: &alive, Scenario: &Status{State: "running", StepID: "cycle"}})
	}
	rows[2].Scenario = &Status{State: "completed", StepID: "cycle", CompletedSteps: 1, EndFrame: 42}
	if r := Analyze(s, rows, rows); r.Accepted || r.State != "trace_invalid" {
		t.Fatal("unproven respawn accepted", r)
	}
	rows[1].Health = &dead
	if r := Analyze(s, rows, rows); !r.Accepted || len(r.Metrics.ActorLifecycle) != 2 {
		t.Fatal("observed cycle rejected", r)
	}
}
