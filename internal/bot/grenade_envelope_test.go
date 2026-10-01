package bot

import (
	"math"
	"q2coopbot/internal/quake"
	"testing"
)

func TestGrenadeContinuousJitterInsideEnvelope(t *testing.T) {
	// An oblique basis and interior jitter values, not just the nine samples.
	q := math.Sqrt(.5)
	fwd := quake.Vec3{q, q, 0}
	right := quake.Vec3{q, -q, 0}
	up := quake.Vec3{0, 0, 1}
	start := quake.Vec3{12, -17, 38}
	for tick := 0; tick <= 5; tick++ {
		b := grenadeBallisticBox(start, fwd, right, up, 800, tick)
		seconds := float64(tick) * .1
		for _, uj := range []float64{-10, -7.3, 0, 4.1, 10} {
			for _, rj := range []float64{-10, -2.7, 0, 8.2, 10} {
				for i := range start {
					point := start[i] + seconds*(400*fwd[i]+(200+uj)*up[i]+rj*right[i])
					if i == 2 {
						point -= 800 * .01 * float64(tick*(tick+1)) / 2
					}
					if point < b.lo[i] || point > b.hi[i] {
						t.Fatal("Interior jitter outside envelope", tick, point, b)
					}
				}
			}
		}
	}
}

func TestGrenadeEnvelopeRejectsUnknownMotionAndNeverAuthorizes(t *testing.T) {
	s := quake.Snapshot{Frame: 20, Gravity: 800, Enemies: []quake.Object{{ID: 9, Solid: 8290, Origin: quake.Vec3{80, 80, 0}}}}
	r := grenadeEarlyEnvelope(s, quake.Vec3{}, quake.Vec3{1, 0, 0}, quake.Vec3{0, -1, 0}, quake.Vec3{0, 0, 1})
	if r.Status != "reject_early_blast" || len(r.Contacts) == 0 || r.Authorized || r.GeometryCertified {
		t.Fatal(r)
	}
	s.Enemies = nil
	r = grenadeEarlyEnvelope(s, quake.Vec3{}, quake.Vec3{1, 0, 0}, quake.Vec3{0, -1, 0}, quake.Vec3{0, 0, 1})
	if r.Status != "unproven" || r.Authorized || r.GeometryCertified {
		t.Fatal("Absence of contacts claimed safe", r)
	}
	s.LastTeammate = &quake.Vec3{80, 80, 0}
	if r = grenadeEarlyEnvelope(s, quake.Vec3{}, quake.Vec3{1, 0, 0}, quake.Vec3{0, -1, 0}, quake.Vec3{0, 0, 1}); r.Status != "reject_unseen_teammate" {
		t.Fatal(r)
	}
}

func TestGrenadeBoxDistanceAndIntersection(t *testing.T) {
	a := grenadeBox{quake.Vec3{1, 2, 3}, quake.Vec3{4, 5, 6}}
	if grenadeBoxDistance(quake.Vec3{1, 2, 3}, a) != 0 || grenadeBoxDistance(quake.Vec3{0, 1, 2}, a) != math.Sqrt(3) {
		t.Fatal("Incorrect blast distance")
	}
	if _, ok := grenadeBoxIntersection(a, grenadeBox{quake.Vec3{5, 2, 3}, quake.Vec3{6, 5, 6}}); ok {
		t.Fatal("Disjoint boxes intersect")
	}
}
