package learningenv

import (
	"q2coopbot/internal/policy"
	"testing"
)

func TestDeathStopNativeTerminal(t *testing.T) {
	for _, name := range []string{"end_frame", "begin_frame", "late_receipt", "missing", "wrong_actor", "wrong_generation", "later_life", "alive", "gap", "unclosed", "outside_window", "not_terminal", "recovery", "health_mismatch", "sequence", "earlier_death"} {
		t.Run(name, func(t *testing.T) {
			id := policy.Identity{Map: "base1", Spawncount: 42, Actor: 1, Connection: 1, Life: 1, Frame: 10}
			next := policy.Observation{Identity: id, Health: -1}
			next.Identity.Frame = 11
			s := &Step{Observation: policy.Observation{Identity: id, Health: 5}, Next: &next, Terminal: true, Reason: "observed_death", ClientSequence: 7,
				Execution: &Execution{Matched: true, WindowExclusive: true}, Native: &NativeStep{Spawncount: 42, Actor: 1, BeginFrame: 10, EndFrame: 11, Sequence: 7, DamageIndexes: []int{0}}}
			g := DeathStop{Version: "combat_first_life_death_stop_v1", Reason: "combat_first_life_death", Map: "base1", Spawncount: 42, Actor: 1, DeathFrame: 11, ObservedFrame: 11, Health: -1}
			r := &CombatRelease{Spawncount: 42, Frame: 9}
			events := []DamageEvent{{Map: "base1", Spawncount: 42, Frame: 11, Target: 1, TargetClass: "player", HealthBefore: 5, HealthAfter: -1}}
			valid := false
			switch name {
			case "end_frame":
				valid = true
			case "begin_frame":
				g.DeathFrame = 10
				events[0].Frame = 10
				valid = true
			case "late_receipt":
				g.ObservedFrame = 13
				g.Health = -2
				valid = true
			case "missing":
				events = nil
			case "wrong_actor":
				events[0].Target = 2
			case "wrong_generation":
				g.Spawncount++
			case "later_life":
				next.Identity.Life = 2
			case "alive":
				next.Health = 1
			case "gap":
				next.Identity.Frame++
			case "unclosed":
				s.Next = nil
			case "outside_window":
				s.Native.DamageIndexes = nil
			case "not_terminal":
				s.Terminal = false
			case "recovery":
				s.Execution.RecoveryCommands = 1
			case "health_mismatch":
				g.Health = -2
			case "sequence":
				s.Native.Sequence++
			case "earlier_death":
				events = append([]DamageEvent{{Map: "base1", Spawncount: 42, Frame: 9, Target: 1, TargetClass: "player", HealthBefore: 1, HealthAfter: 0}}, events...)
				s.Native.DamageIndexes = []int{1}
			}
			if err := VerifyDeathStop(g, r, events, s); (err == nil) != valid {
				t.Fatalf("valid=%v err=%v", valid, err)
			}
		})
	}
}
