package bot

import (
	"math"
	"q2coopbot/internal/quake"
	"testing"
)

func TestGrenadeBounceReachContainsDifferentNormalsAndClamping(t *testing.T) {
	seed := grenadeBox{quake.Vec3{}, quake.Vec3{}}
	bound := grenadeBounceReach(seed, 0, 8, 800)
	for _, normal := range []quake.Vec3{{0, 0, 1}, {-1, 0, 0}, {0, -1, 0}, {math.Sqrt(.5), 0, math.Sqrt(.5)}} {
		for _, speed := range []float64{400, 3000} {
			tick := 0
			trace := func(from, to quake.Vec3) quake.PointTrace {
				point := from
				for i := range point {
					point[i] += (to[i] - from[i]) * .5
				}
				return quake.PointTrace{Valid: true, Fraction: .5, End: point, Normal: normal}
			}
			simulateGrenade(quake.Vec3{}, quake.Vec3{speed, speed, 200}, .9, 800, trace, nil, func(point quake.Vec3, _ int) {
				tick++
				b := bound.Ranges[tick]
				for i := range point {
					if point[i] < b.Min[i] || point[i] > b.Max[i] {
						t.Fatal("Bounce escaped envelope", normal, speed, tick, point, b)
					}
				}
			})
		}
	}
	if bound.Authorized || bound.GeometryCertified {
		t.Fatal("Coarse envelope authorized throw")
	}
	if grenadeBounceReach(grenadeBox{quake.Vec3{1, 0, 0}, quake.Vec3{}}, 0, 2, 800) != nil || grenadeBounceReach(seed, 0, 40, 800) != nil {
		t.Fatal("Invalid bound input accepted")
	}
}

func TestGrenadeNativeBounceReachRejectsOutsideObservation(t *testing.T) {
	c := GrenadeCalibration{Launch: &GrenadeLaunchCheck{ExpectedExplosionFrame: 14}, Points: []GrenadeComparison{
		{Frame: 10, Predicted: quake.Vec3{}},
		{Frame: 11, Predicted: quake.Vec3{10, 0, 0}, Observed: quake.Vec3{10, 0, 0}, Bounces: 1},
		{Frame: 12, Predicted: quake.Vec3{15, 0, 0}, Observed: quake.Vec3{15, 0, 0}, Bounces: 1},
		{Frame: 13, Predicted: quake.Vec3{20, 0, 0}, Observed: quake.Vec3{20, 0, 0}, Bounces: 2},
	}}
	if err := checkGrenadeBounceReach(&c, 800); err != nil {
		t.Fatal(err)
	}
	c.Points[2].Observed[0] = 1000
	if checkGrenadeBounceReach(&c, 800) == nil {
		t.Fatal("Corrupt post-bounce point accepted")
	}
}

func TestGrenadeSpeedNormLimitBeforeGravity(t *testing.T) {
	f := grenadeFlight(quake.Vec3{}, quake.Vec3{3000, 4000, 0}, .2, 800, emptyGrenadeTrace, nil)
	if f.End != (quake.Vec3{120, 160, -8}) {
		t.Fatal("Native speed/gravity order disagrees", f)
	}
	if grenadeBounceReach(grenadeBox{}, 0, 2, math.NaN()) != nil {
		t.Fatal("NaN gravity accepted")
	}
}
