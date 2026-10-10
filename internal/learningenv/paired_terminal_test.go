package learningenv

import (
	"q2coopbot/internal/harness"
	"q2coopbot/internal/policy"
	"testing"
)

func pairedDeathFixture(deadRole int) (*NativePairs, [2][]harness.Trace, []DamageEvent) {
	n := &NativePairs{Release: &CombatRelease{Spawncount: 42, Frame: 100, Seed: 123}}
	var traces [2][]harness.Trace
	for i := 0; i < 2; i++ {
		var pair NativePair
		for role := 0; role < 2; role++ {
			h := int16(100)
			if i == 1 && role == deadRole {
				h = 0
			}
			pair.Steps[role] = NativeStep{Spawncount: 42, Actor: role + 1, BeginFrame: 100 + i, EndFrame: 101 + i, Sequence: uint32(i + 1)}
			pair.Commands[role] = harness.AppliedCommand{}
			traces[role] = append(traces[role], harness.Trace{Connection: 1, Generation: 42, Frame: 100 + i, SelfEntity: role + 1, ClientSequence: uint32(i + 1), Map: "base1", Health: &h})
		}
		n.Pairs = append(n.Pairs, pair)
	}
	n.Pairs[0].Steps[0].DamageIndexes = []int{0}
	n.Pairs[0].Steps[1].DamageIndexes = []int{0}
	e := []DamageEvent{{Spawncount: 42, Map: "base1", Frame: 100, Target: deadRole + 1, TargetClass: "player", HealthBefore: 100, HealthAfter: 0}}
	return n, traces, e
}

func TestPairedTerminalMarksSurvivorAndDeadRole(t *testing.T) {
	n, traces, events := pairedDeathFixture(1)
	b, err := n.FirstDeathBoundary(traces, events)
	if err != nil {
		t.Fatal(err)
	}
	for role := 0; role < 2; role++ {
		id := policy.Identity{Map: "base1", Connection: 1, Life: 1, Spawncount: 42, Actor: role + 1, Frame: 100}
		next := policy.Observation{Identity: id, Health: b.HealthAfter[role]}
		next.Identity.Frame++
		s := Step{Observation: policy.Observation{Identity: id, Health: 100}, Next: &next, Native: &n.Pairs[0].Steps[role], Execution: &Execution{Matched: true, WindowExclusive: true}, ClientSequence: 1}
		if err := b.MarkTerminal(&s, role); err != nil || !s.Terminal || s.Truncated || s.JointTerminal != b {
			t.Fatal(s, err)
		}
		s.Next.Health = 7
		if b.MarkTerminal(&s, role) == nil {
			t.Fatal("changed health terminal accepted")
		}
	}
}

func TestPairedScriptedPeerDoesNotBypassOrdinaryOverride(t *testing.T) {
	c := policy.Capture{LimitReason: "test_idle", Observation: policy.Observation{Identity: policy.Identity{Frame: 100}}}
	ordinary := Assembler{}
	paired := Assembler{KeepScriptedPeerFromFrame: 100}
	if !ordinary.harnessOverride(c) || paired.harnessOverride(c) {
		t.Fatal("scripted peer filter")
	}
	c.Observation.Identity.Frame = 99
	if !paired.harnessOverride(c) {
		t.Fatal("scripted setup retained before release")
	}
	c.Observation.Identity.Frame = 100
	c.LimitReason = "test_teleport_settling"
	if !paired.harnessOverride(c) {
		t.Fatal("teleport setup became a usable step")
	}
}

func TestPairedDeathBoundaryBothRoles(t *testing.T) {
	for role := 0; role < 2; role++ {
		n, traces, events := pairedDeathFixture(role)
		b, err := n.FirstDeathBoundary(traces, events)
		if err != nil || b == nil || b.EndFrame != 101 || b.BeginFrame != 100 || b.HealthAfter[role] != 0 || b.HealthAfter[1-role] != 100 {
			t.Fatal(b, err)
		}
	}
	n, traces, events := pairedDeathFixture(0)
	zero := int16(0)
	traces[1][1].Health = &zero
	events = append(events, DamageEvent{Spawncount: 42, Map: "base1", Frame: 101, Target: 2, TargetClass: "player", HealthBefore: 100, HealthAfter: 0})
	n.Pairs[0].Steps[0].DamageIndexes = append(n.Pairs[0].Steps[0].DamageIndexes, 1)
	b, err := n.FirstDeathBoundary(traces, events)
	if err != nil || b == nil || len(b.DeathEventIndexes) != 2 {
		t.Fatal(b, err)
	}
}

func TestPairedFinalObservationClosesLastNativeTick(t *testing.T) {
	n, traces, events := pairedDeathFixture(1)
	n.Pairs = n.Pairs[:1]
	for role := 0; role < 2; role++ {
		traces[role][1].TerminalObservationOnly = true
		traces[role][1].ClientSequence = 0
	}
	b, err := n.FirstDeathBoundary(traces, events)
	if err != nil || b == nil || b.EndFrame != 101 {
		t.Fatal(b, err)
	}
	traces[1][1].Frame++
	if _, err := n.FirstDeathBoundary(traces, events); err == nil {
		t.Fatal("unaligned final observation accepted")
	}
}

func TestPairedDeathBoundaryRejectsUnprovenHealth(t *testing.T) {
	for _, scenario := range []string{"missing_next", "alive_after_death", "unknown_health", "outside_tick", "bad_index", "changed_map", "changed_native_command", "already_dead"} {
		t.Run(scenario, func(t *testing.T) {
			n, traces, events := pairedDeathFixture(1)
			switch scenario {
			case "missing_next":
				n.Pairs = n.Pairs[:1]
				traces[0] = traces[0][:1]
				traces[1] = traces[1][:1]
			case "alive_after_death":
				h := int16(10)
				traces[1][1].Health = &h
			case "unknown_health":
				traces[0][1].Health = nil
			case "outside_tick":
				events[0].Frame = 103
			case "bad_index":
				n.Pairs[0].Steps[0].DamageIndexes = []int{9}
			case "changed_map":
				traces[0][1].Map = "base2"
			case "changed_native_command":
				traces[1][1].Command.Forward = 1
			case "already_dead":
				h := int16(0)
				traces[1][0].Health = &h
			}
			if _, err := n.FirstDeathBoundary(traces, events); err == nil {
				t.Fatal("unproven boundary accepted")
			}
		})
	}
}
