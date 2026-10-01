package quake

import (
	"math"
	"testing"
)

func TestProjectileTraceBrushEntry(t *testing.T) {
	c := &CollisionMap{worldBrushes: []int{0}, brushes: []bspBrush{{0, 6, 1}}, sides: []uint16{0, 1, 2, 3, 4, 5}, planes: []bspPlane{
		{Vec3{1, 0, 0}, 20}, {Vec3{-1, 0, 0}, -10}, {Vec3{0, 1, 0}, 5}, {Vec3{0, -1, 0}, 5}, {Vec3{0, 0, 1}, 5}, {Vec3{0, 0, -1}, 5},
	}}
	tr := c.traceProjectile(Vec3{}, Vec3{40, 0, 0})
	if !tr.Valid || tr.StartSolid || math.Abs(tr.End[0]-9.96875) > 1e-8 || tr.Normal != (Vec3{-1, 0, 0}) {
		t.Fatalf("%+v", tr)
	}
	if tr = c.traceProjectile(Vec3{15, 0, 0}, Vec3{40, 0, 0}); !tr.StartSolid {
		t.Fatal(tr)
	}
	if tr = c.traceProjectile(Vec3{0, 10, 0}, Vec3{40, 10, 0}); tr.Fraction != 1 {
		t.Fatal(tr)
	}
	c.sideSurfaces = make([]groundSurface, 6)
	c.sideSurfaces[1] = groundSurface{true, 4}
	if tr = c.traceProjectile(Vec3{}, Vec3{40, 0, 0}); !tr.Sky {
		t.Fatal(tr)
	}
	if tr = c.traceProjectile(Vec3{math.NaN(), 0, 0}, Vec3{}); tr.Valid {
		t.Fatal(tr)
	}
	if tr = (*MapInfo)(nil).TraceProjectile(Vec3{}, Vec3{}); tr.Valid {
		t.Fatal(tr)
	}
}
