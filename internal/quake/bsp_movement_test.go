package quake

import "testing"

func testBoxBrush(mins, maxs Vec3, first int) ([]bspPlane, bspBrush) {
	planes := []bspPlane{
		{Vec3{1, 0, 0}, maxs[0]}, {Vec3{-1, 0, 0}, -mins[0]},
		{Vec3{0, 1, 0}, maxs[1]}, {Vec3{0, -1, 0}, -mins[1]},
		{Vec3{0, 0, 1}, maxs[2]}, {Vec3{0, 0, -1}, -mins[2]},
	}
	return planes, bspBrush{first: first, count: len(planes), contents: 1}
}

func TestPlayerMoveClearAndGroundDrop(t *testing.T) {
	floor, floorBrush := testBoxBrush(Vec3{-100, -100, -20}, Vec3{100, 100, 0}, 0)
	wall, wallBrush := testBoxBrush(Vec3{40, -100, 0}, Vec3{60, 100, 100}, len(floor))
	c := &CollisionMap{planes: append(floor, wall...), brushes: []bspBrush{floorBrush, wallBrush}, worldBrushes: []int{0, 1}}
	for i := range c.planes {
		c.sides = append(c.sides, uint16(i))
	}
	m := &MapInfo{collision: c}
	if !m.PlayerMoveClear(Vec3{0, 0, 24}, Vec3{20, 0, 24}) {
		t.Fatal("clear movement over floor was blocked")
	}
	if m.PlayerMoveClear(Vec3{0, 0, 24}, Vec3{40, 0, 24}) {
		t.Fatal("standing player hull crossed a wall")
	}
	if drop, ok := m.GroundDrop(Vec3{20, 0, 24}, 24); !ok || drop > 1 {
		t.Fatalf("floor support missing: drop=%f ok=%t", drop, ok)
	}
	if _, ok := m.GroundDrop(Vec3{120, 0, 24}, 24); ok {
		t.Fatal("unsupported position was marked grounded")
	}
	if got := m.GroundMoveHazard(nil, Vec3{0, 0, 24}, 40, 0); got != "static_hull_blocked" {
		t.Fatalf("wall hazard=%q", got)
	}
	if got := m.GroundMoveHazard(nil, Vec3{90, 0, 24}, 40, 0); got != "no_ground_support" {
		t.Fatalf("edge hazard=%q", got)
	}
	n := &Navigator{Areas: []Area{{}, {Min: Vec3{110, -20, 0}, Max: Vec3{140, 20, 40}, Flags: 1}}}
	if got := m.GroundMoveHazard(n, Vec3{90, 0, 24}, 40, 0); got != "" {
		t.Fatalf("AAS grounded support was ignored: %q", got)
	}
}

func TestGroundedNearDoesNotTrustDistantRouteFallback(t *testing.T) {
	n := &Navigator{Areas: []Area{{}, {Min: Vec3{0, 0, 0}, Max: Vec3{20, 20, 40}, Flags: 1}}}
	if !n.GroundedNear(Vec3{21, 10, 24}) {
		t.Fatal("one-unit AAS rounding gap was rejected")
	}
	if n.GroundedNear(Vec3{30, 10, 24}) {
		t.Fatal("distant grounded area was used as support")
	}
}
