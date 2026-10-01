package bot

import (
	"math"
	"os"
	"q2coopbot/internal/quake"
	"testing"
)

func TestGrenadeCandidateOnBase1(t *testing.T) {
	root := os.Getenv("Q2_SEARCH_SCAN_ROOT")
	if root == "" {
		t.Skip("Q2_SEARCH_SCAN_ROOT not set")
	}
	g, err := quake.LoadMap(root, "base1")
	if err != nil {
		t.Fatal(err)
	}
	p := Planner{World: World{Geometry: &g}}
	ally := quake.Vec3{-945, 1292, 344.125}
	s := quake.Snapshot{Self: quake.Vec3{-937, 1149, 344.125}, Gravity: 800, GunFrame: 16, Teammate: &ally}
	r := p.predictGrenadeCandidate(s)
	if r.Reason != "not_authorized" || len(r.Samples) != 9 {
		t.Fatal(r)
	}
	for _, f := range r.Samples {
		if f.Event != "fuse" || f.Bounces < 1 || f.Risk != "teammate_reachable_blast" {
			t.Fatalf("%+v", f)
		}
	}
	s.GunFrame = 11
	if r = p.predictGrenadeCandidate(s); r.Reason != "armed_fuse_unknown" || len(r.Samples) != 0 {
		t.Fatal(r)
	}
	s.GunFrame = 16
	s.Movers = []quake.Mover{{Model: 99999}}
	r = p.predictGrenadeCandidate(s)
	for _, f := range r.Samples {
		if f.Event != "unknown_collision" {
			t.Fatal(f)
		}
	}
}

func emptyGrenadeTrace(from, to quake.Vec3) quake.PointTrace {
	return quake.PointTrace{Fraction: 1, End: to, Valid: true}
}

func TestGrenadeNativeTickAndFuse(t *testing.T) {
	r := grenadeFlight(quake.Vec3{}, quake.Vec3{400, 0, 200}, 0.3, 800, emptyGrenadeTrace, nil)
	// Think happens before movement on the third tick, no leftover motion.
	if r.Event != "fuse" || r.Seconds != 0.3 && math.Abs(r.Seconds-0.3) > 1e-9 || r.End != (quake.Vec3{80, 0, 16}) {
		t.Fatalf("%+v", r)
	}
}

func TestGrenadeBounceDiscardsRemainder(t *testing.T) {
	n := 0
	trace := func(from, to quake.Vec3) quake.PointTrace {
		n++
		if n == 1 {
			return quake.PointTrace{Fraction: 0.5, End: quake.Vec3{20, 0, 0}, Normal: quake.Vec3{-1, 0, 0}, Valid: true}
		}
		return emptyGrenadeTrace(from, to)
	}
	r := grenadeFlight(quake.Vec3{}, quake.Vec3{400, 0, 80}, 0.3, 800, trace, nil)
	if r.Event != "fuse" || r.Bounces != 1 || r.End != (quake.Vec3{0, 0, -8}) {
		t.Fatalf("%+v", r)
	}
}

func TestGrenadeLivingContactBeforeWall(t *testing.T) {
	b := grenadeBody{origin: quake.Vec3{20, 0, 0}, mins: quake.Vec3{-1, -1, -1}, maxs: quake.Vec3{1, 1, 1}}
	r := grenadeFlight(quake.Vec3{}, quake.Vec3{400, 0, 80}, 3, 800, emptyGrenadeTrace, []grenadeBody{b})
	if r.Event != "damageable_contact" || r.Seconds != 0.1 || r.End[0] != 19 {
		t.Fatalf("%+v", r)
	}
	wall := func(from, to quake.Vec3) quake.PointTrace {
		return quake.PointTrace{Fraction: 0.2, End: quake.Vec3{8, 0, 0}, Normal: quake.Vec3{-1, 0, 0}, Valid: true}
	}
	r = grenadeFlight(quake.Vec3{}, quake.Vec3{400, 0, 80}, 0.2, 800, wall, []grenadeBody{b})
	if r.Event != "fuse" || r.Bounces != 1 {
		t.Fatalf("Contact through wall: %+v", r)
	}
}

func TestGrenadeUnknownAndSky(t *testing.T) {
	for _, tr := range []quake.PointTrace{{}, {Valid: true, StartSolid: true}, {Valid: true, Fraction: 0.5, Sky: true}} {
		r := grenadeFlight(quake.Vec3{}, quake.Vec3{100, 0, 0}, 3, 800, func(_, _ quake.Vec3) quake.PointTrace { return tr }, nil)
		if r.Event != "unknown_collision" && r.Event != "sky" {
			t.Fatalf("%+v", r)
		}
	}
	for _, f := range []float64{0, -1, math.NaN(), math.Inf(1), 4} {
		if r := grenadeFlight(quake.Vec3{}, quake.Vec3{}, f, 800, emptyGrenadeTrace, nil); r.Event != "invalid_input" {
			t.Fatal(r)
		}
	}
	var p Planner
	if r := p.predictGrenadeCandidate(quake.Snapshot{GunFrame: 11}); len(r.Samples) != 0 {
		t.Fatal(r)
	}
}
