package bot

import (
	"q2coopbot/internal/policy"
	"q2coopbot/internal/quake"
	"testing"
)

func TestEngagementOcclusionAndObservedDefeat(t *testing.T) {
	e := combatEngagement{}
	o := policy.Observation{Identity: policy.Identity{Map: "base1", Connection: 1, Spawncount: 1, Actor: 1, Life: 1, Frame: 20}, Enemies: []policy.Enemy{{ID: 7, Class: "monster_parasite"}, {ID: 8, Class: "monster_gunner"}}}
	if a, c, _ := e.observe(o, nil); !a || c {
		t.Fatal("visible fight not active")
	}
	o.Identity.Frame++
	o.Enemies = nil
	if a, c, _ := e.observe(o, []quake.Object{{ID: 8, Class: "monster_gunner"}}); !a || !c {
		t.Fatal("remaining occluded threat handed to rules")
	}
	if len(o.Enemies) != 0 {
		t.Fatal("occluded enemy invented")
	}
	o.Identity.Frame++
	if a, c, r := e.observe(o, []quake.Object{{ID: 7, Class: "monster_parasite"}}); a || c || r != "system2_noncombat" {
		t.Fatal("observed defeat did not finish", a, c, r)
	}
}

func TestEngagementGraceUsesFramesAndCannotCrossReset(t *testing.T) {
	o := policy.Observation{Identity: policy.Identity{Map: "base1", Connection: 1, Spawncount: 1, Actor: 1, Life: 1, Frame: 20}, Enemies: []policy.Enemy{{ID: 7, Class: "monster_parasite"}}}
	e := combatEngagement{}
	e.observe(o, nil)
	o.Enemies = nil
	for f := 21; f <= 50; f++ {
		o.Identity.Frame = f
		if a, c, _ := e.observe(o, nil); !a || !c {
			t.Fatal("grace ended early", f)
		}
	}
	o.Identity.Frame = 51
	if a, c, r := e.observe(o, nil); a || c || r != "combat_visibility_timeout" {
		t.Fatal("timeout not explicit", a, c, r)
	}
	for _, change := range []func(*policy.Identity){func(i *policy.Identity) { i.Life++ }, func(i *policy.Identity) { i.Connection++ }, func(i *policy.Identity) { i.Spawncount++ }, func(i *policy.Identity) { i.Map = "base2" }, func(i *policy.Identity) { i.Frame += 2 }, func(i *policy.Identity) { i.Frame-- }} {
		original := o
		original.Enemies = []policy.Enemy{{ID: 7, Class: "monster_parasite"}}
		e = combatEngagement{}
		e.observe(original, nil)
		original.Enemies = nil
		change(&original.Identity)
		if a, _, _ := e.observe(original, nil); a {
			t.Fatal("ownership crossed reset/gap")
		}
	}
}

func TestEngagementThreatMemoryHorizon(t *testing.T) {
	o := policy.Observation{Identity: policy.Identity{Map: "base1", Connection: 1, Spawncount: 1, Actor: 1, Life: 1, Frame: 20}, Enemies: []policy.Enemy{{ID: 7, Class: "monster_soldier"}}}
	e := combatEngagement{}
	e.observe(o, nil, policy.ThreatMemoryFrames)
	o.Enemies = nil
	for frame := 21; frame <= 220; frame++ {
		o.Identity.Frame = frame
		if active, continuation, _ := e.observe(o, nil, policy.ThreatMemoryFrames); !active || !continuation {
			t.Fatal("memory search handed off early", frame)
		}
	}
	o.Identity.Frame++
	if active, _, reason := e.observe(o, nil, policy.ThreatMemoryFrames); active || reason != "combat_visibility_timeout" {
		t.Fatal("unbounded continuation")
	}
}
