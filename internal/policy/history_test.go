package policy

import (
	"q2coopbot/internal/quake"
	"testing"
)

func historyObservation(frame int) Observation {
	return Observation{Version: ObservationVersion, Identity: Identity{Life: 1, Map: "base1", Connection: 1, Spawncount: 2, Actor: 1, Frame: frame}, Health: 100,
		Position: quake.Vec3{float64(frame) * 2, 0, 0}, Enemies: []Enemy{{ID: 67, Class: "monster_parasite", Relative: quake.Vec3{100, 0, 0}}}}
}

func TestHistoryUsesGameTimeAndOwnDisplacement(t *testing.T) {
	h := History{}
	a := historyObservation(10)
	h.Enrich(&a)
	if len(a.History) != 0 || a.Enemies[0].Velocity != nil || a.Enemies[0].Track == nil {
		t.Fatal(a)
	}
	track := *a.Enemies[0].Track
	for frame := 11; frame <= 16; frame++ {
		b := historyObservation(frame)
		h.Enrich(&b)
		if *b.Enemies[0].Track != track || b.Enemies[0].Velocity[0] != 20 || len(b.History) != min(frame-10, HistoryFrames) {
			t.Fatal(b)
		}
		if b.History[len(b.History)-1].Identity.Frame != frame-1 {
			t.Fatal("future/history frame leak")
		}
		// A provider mutating its inputs cannot poison subsequent history.
		*b.Enemies[0].Track = 999
		*b.History[0].Enemies[0].Track = 888
		b.History[0].Enemies[0].Relative[0] = 9999
	}
}

func TestHistoryDoesNotTrackThroughOcclusionReuseOrBoundary(t *testing.T) {
	h := History{}
	a := historyObservation(10)
	h.Enrich(&a)
	original := *a.Enemies[0].Track
	b := historyObservation(11)
	b.Enemies = nil
	h.Enrich(&b)
	c := historyObservation(12)
	h.Enrich(&c)
	if *c.Enemies[0].Track == original || c.Enemies[0].Velocity != nil {
		t.Fatal("velocity crossed occlusion")
	}
	d := historyObservation(13)
	d.Enemies[0].Class = "monster_gunner"
	h.Enrich(&d)
	if *d.Enemies[0].Track == *c.Enemies[0].Track || d.Enemies[0].Velocity != nil {
		t.Fatal("reused class inherited track")
	}
	for _, change := range []func(*Observation){
		func(o *Observation) { o.Identity.Frame += 2 }, func(o *Observation) { o.Identity.Life++ },
		func(o *Observation) { o.Identity.Spawncount++ }, func(o *Observation) { o.Identity.Connection++ },
		func(o *Observation) { o.AgeMS = 301 }, func(o *Observation) { o.Health = 0 },
	} {
		h = History{}
		a = historyObservation(10)
		h.Enrich(&a)
		b = historyObservation(11)
		change(&b)
		h.Enrich(&b)
		if len(b.History) != 0 || b.Enemies[0].Velocity != nil {
			t.Fatal("history crossed boundary", b)
		}
	}
}
