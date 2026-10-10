package policy

import (
	"q2coopbot/internal/quake"
	"testing"
)

func TestThreatMemoryOcclusionDoesNotReadHiddenPositions(t *testing.T) {
	yes := true
	o := Observation{Identity: Identity{Map: "base1", Connection: 1, Spawncount: 1, Actor: 1, Life: 1, Frame: 10}, Health: 100, Position: quake.Vec3{10, 0, 0}, Enemies: []Enemy{{ID: 7, Class: "monster_soldier", Relative: quake.Vec3{100, 20, 0}, ClearShot: &yes, Velocity: &quake.Vec3{40, 0, 0}}}}
	m := ThreatMemory{}
	m.Enrich(&o, nil)
	o.Enemies = nil
	o.Identity.Frame++
	o.Position[0] += 5
	m.Enrich(&o, nil)
	if len(o.RememberedThreats) != 1 || o.RememberedThreats[0].Relative != (quake.Vec3{95, 20, 0}) || o.RememberedThreats[0].AgeFrames != 1 || len(o.Enemies) != 0 {
		t.Fatal(o.RememberedThreats)
	}
	// No velocity extrapolation or fabricated current target. Returned vectors
	// cannot mutate the stored observation.
	o.RememberedThreats[0].Velocity[0] = 999
	o.Identity.Frame++
	m.Enrich(&o, nil)
	if o.RememberedThreats[0].Velocity[0] != 40 {
		t.Fatal("memory aliases output")
	}
	o.Identity.Frame++
	m.Enrich(&o, []quake.Object{{ID: 7, Class: "monster_soldier"}})
	if len(o.RememberedThreats) != 0 {
		t.Fatal("observed death retained")
	}
}

func TestThreatMemoryExpiresAndResets(t *testing.T) {
	yes := true
	o := Observation{Identity: Identity{Map: "base1", Connection: 1, Spawncount: 1, Actor: 1, Life: 1, Frame: 10}, Health: 100, Enemies: []Enemy{{ID: 7, Class: "monster_soldier", ClearShot: &yes}}}
	m := ThreatMemory{}
	m.Enrich(&o, nil)
	o.Enemies = nil
	for frame := 11; frame <= 210; frame++ {
		o.Identity.Frame = frame
		m.Enrich(&o, nil)
		if len(o.RememberedThreats) != 1 {
			t.Fatal("early expiration", frame)
		}
	}
	o.Identity.Frame++
	m.Enrich(&o, nil)
	if len(o.RememberedThreats) != 0 {
		t.Fatal("stale threat retained")
	}
	for _, change := range []func(*Observation){func(o *Observation) { o.Identity.Frame += 2 }, func(o *Observation) { o.Identity.Life++ }, func(o *Observation) { o.Identity.Map = "base2" }, func(o *Observation) { o.Identity.Connection++ }, func(o *Observation) { o.Identity.Spawncount++ }, func(o *Observation) { o.AgeMS = 301 }, func(o *Observation) { o.Health = 0 }} {
		s := o
		s.Enemies = []Enemy{{ID: 7, Class: "monster_soldier", ClearShot: &yes}}
		m = ThreatMemory{}
		m.Enrich(&s, nil)
		s.Enemies = nil
		s.Identity.Frame++
		change(&s)
		m.Enrich(&s, nil)
		if len(s.RememberedThreats) != 0 {
			t.Fatal("memory crossed reset/gap/stale/death")
		}
	}
}
