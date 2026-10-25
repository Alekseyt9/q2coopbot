package quake

import "testing"

func TestStaticPointContentsDistinguishesWaterLavaAndEmpty(t *testing.T) {
	c := &CollisionMap{}
	for i, contents := range []int{32, 8, 16} {
		lo, hi := Vec3{float64(i * 100), 0, 0}, Vec3{float64(i*100 + 50), 50, 50}
		planes, b := testBoxBrush(lo, hi, len(c.sides))
		b.contents = contents
		for _, p := range planes {
			c.sides = append(c.sides, uint16(len(c.planes)))
			c.planes = append(c.planes, p)
		}
		c.worldBrushes = append(c.worldBrushes, len(c.brushes))
		c.brushes = append(c.brushes, b)
	}
	m := MapInfo{collision: c}
	for _, tc := range []struct {
		point Vec3
		want  int
	}{{Vec3{10, 10, 10}, 32}, {Vec3{110, 10, 10}, 8}, {Vec3{210, 10, 10}, 16}, {Vec3{-100, 0, 0}, 0}} {
		got, ok := m.StaticPointContents(tc.point)
		if !ok || got != tc.want {
			t.Fatal(tc, got, ok)
		}
	}
	if _, ok := (&MapInfo{}).StaticPointContents(Vec3{}); ok {
		t.Fatal("missing collision treated as empty")
	}
}
