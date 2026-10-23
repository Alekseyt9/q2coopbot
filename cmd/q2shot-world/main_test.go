package main

import (
	"math"
	"q2coopbot/internal/quake"
	"testing"
)

func TestObservedShotRayCoordinatesAndParallelSlabs(t *testing.T) {
	x := make([]float64, 185)
	x[7] = 1
	x[9] = 1
	x[99] = 1
	x[101] = .5
	x[147] = 1
	x[148] = .25
	x[149] = -.375
	x[150] = .5
	p := quake.Vec3{100, 200, 300}
	r, err := observedRay(x, p)
	if err != nil || !r.Known || !r.Intersects || r.Muzzle != (quake.Vec3{100, 192, 314}) || math.Abs(r.Near-240) > 1e-9 {
		t.Fatalf("%+v %v", r, err)
	}
	center, err := observedRayForHand(x, p, 2)
	if err != nil || center.Muzzle != (quake.Vec3{100, 200, 314}) || !center.Intersects {
		t.Fatal("native hand2 contract", center, err)
	}
	// Rotate both observed view and yaw-local enemy by90deg: translated world
	// muzzle rotates with Quake's right-vector convention, bbox entry unchanged.
	x[6] = 1
	x[7] = 0
	r, err = observedRay(x, p)
	if err != nil || !r.Intersects || math.Abs(r.Muzzle[0]-108) > 1e-9 || math.Abs(r.Muzzle[1]-200) > 1e-9 || math.Abs(r.Near-240) > 1e-9 {
		t.Fatalf("%+v %v", r, err)
	}
	_, _, ok := slabInterval(quake.Vec3{}, quake.Vec3{1, 0, 0}, quake.Vec3{10, 1, -1}, quake.Vec3{20, 2, 1})
	if ok {
		t.Fatal("parallel ray outside bbox must miss")
	}
	near, far, ok := slabInterval(quake.Vec3{}, quake.Vec3{1, 0, 0}, quake.Vec3{-1, -1, -1}, quake.Vec3{1, 1, 1})
	if !ok || near != 0 || far != 1 {
		t.Fatal(near, far, ok)
	}
	x[151] = 1
	r, err = observedRay(x, p)
	if err != nil || r.Known {
		t.Fatal(r, err)
	}
}
