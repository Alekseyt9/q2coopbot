package quake

import (
	"math"

	"testing"
)

func TestCommonSurfaceIncludesContinuousContactFamily(t *testing.T) {
	planes, brush := testBoxBrush(Vec3{10, -20, -20}, Vec3{30, 20, 20}, 0)
	c := &CollisionMap{planes: planes, brushes: []bspBrush{brush}, worldBrushes: []int{0}, sides: []uint16{0, 1, 2, 3, 4, 5}, sideSurfaces: make([]groundSurface, 6)}
	for i := range c.sideSurfaces {
		c.sideSurfaces[i].known = true
	}
	a, b, d, e := Vec3{0, -2, -3}, Vec3{1, 2, 3}, Vec3{15, -4, -5}, Vec3{17, 4, 5}
	r := c.projectileCommonSurface(a, b, d, e)
	if r == nil || r.Normal != (Vec3{-1, 0, 0}) || math.Abs(r.Min[0]-9.968749) > 1e-8 {
		t.Fatal(r)
	}
	// Independent endpoints vary throughout their boxes, not nine fixed rays.
	for i := 0; i <= 100; i++ {
		f := float64(i) / 100
		from, to := Vec3{f, -2 + 4*f, -3 + 6*f}, Vec3{15 + 2*f, 4 - 8*f, 5 - 10*f}
		tr := c.traceProjectile(from, to)
		for axis := range tr.End {
			if tr.End[axis] < r.Min[axis] || tr.End[axis] > r.Max[axis] {
				t.Fatal(tr, r)
			}
		}
	}
	if c.projectileCommonSurface(a, b, Vec3{9, -4, -5}, e) != nil {
		t.Fatal("some trajectories do not reach the wall")
	}
	if c.projectileCommonSurface(a, b, d, Vec3{17, 25, 5}) != nil {
		t.Fatal("edge may select another face")
	}
	c.sideSurfaces[1].flags = 4
	if c.projectileCommonSurface(a, b, d, e) != nil {
		t.Fatal("sky accepted")
	}
	c.sideSurfaces[1].flags = 0
	c.sideSurfaces[1].known = false
	if c.projectileCommonSurface(a, b, d, e) != nil {
		t.Fatal("unknown surface accepted")
	}
	c.sideSurfaces[1].known = true
	// A second possible brush defeats the common-face proof.
	c.worldBrushes = append(c.worldBrushes, 0)
	if c.projectileCommonSurface(a, b, d, e) != nil {
		t.Fatal("multiple brushes accepted")
	}
	if (*MapInfo)(nil).ProjectileCommonSurface(a, b, d, e) != nil {
		t.Fatal("missing BSP")
	}
	a[0] = math.NaN()
	if c.projectileCommonSurface(a, b, d, e) != nil {
		t.Fatal("invalid bounds")
	}
}
