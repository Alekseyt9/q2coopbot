package quake

import (
	"math"
	"testing"
)

func TestRotatedMoverSupportAndHull(t *testing.T) {
	planes, brush := testBoxBrush(Vec3{-40, -8, -8}, Vec3{40, 8, 0}, 0)
	g := &MapInfo{collision: &CollisionMap{planes: planes, brushes: []bspBrush{brush}, sides: []uint16{0, 1, 2, 3, 4, 5}, modelBrushes: [][]int{nil, {0}}}}
	deck := Mover{Model: 1, Origin: Vec3{100, 200, 64}, Angles: Vec3{0, 90, 0}}
	standing := Vec3{100, 230, 88.125}
	if drop, ok := g.MoverFooting(deck, standing, 1); !ok || drop > .5 {
		t.Fatal("rotated deck support missing", drop, ok)
	}
	if !g.MoverHullClear(deck, standing, Vec3{100, 235, 88.125}) {
		t.Fatal("standing movement blocked by supporting deck")
	}
	from, to := Vec3{150, 230, 64}, Vec3{100, 230, 64}
	if g.MoverHullClear(deck, from, to) {
		t.Fatal("rotated deck ignored by hull sweep")
	}
	deck.Angles = Vec3{}
	if _, ok := g.MoverFooting(deck, standing, 1); ok {
		t.Fatal("old deck footprint mistaken for rotated support")
	}
	if !g.MoverHullClear(deck, from, to) {
		t.Fatal("rotation mutated original BSP planes")
	}
	deck.Angles = Vec3{0, 0, 90}
	if g.MoverHullClear(deck, Vec3{100, 228, 64}, Vec3{100, 228, 64}) {
		t.Fatal("geometrically clear hull ignored native transformed-box collision")
	}
	if drop, ok := g.MoverFooting(deck, Vec3{100, 204, 96.125}, 1); !ok || drop > .5 {
		t.Fatal("roll did not rotate supporting face", drop, ok)
	}
	deck.Angles = Vec3{}
	if _, ok := g.MoverFooting(deck, Vec3{100, 204, 96.125}, 1); ok {
		t.Fatal("unrotated deck supported rotated face height")
	}
	deck.Angles[0] = math.NaN()
	if g.MoverHullClear(deck, from, to) {
		t.Fatal("invalid orientation trusted")
	}
}

func TestMoverAnglesDecodeAndDeltaRetention(t *testing.T) {
	r := reader{data: []byte{64, 128, 192}}
	e, err := parseEntity(&r, 7, 0x400|4|8, Entity{})
	if err != nil || e.Angles != (Vec3{90, 180, 270}) || r.pos != 3 {
		t.Fatal("packed angles lost", e, err, r.pos)
	}
	r = reader{data: []byte{32}}
	e, err = parseEntity(&r, 7, 4, e)
	if err != nil || e.Angles != (Vec3{90, 45, 270}) {
		t.Fatal("delta overwrote unchanged axes", e, err)
	}
	r = reader{}
	if _, err = parseEntity(&r, 7, 4, e); err == nil {
		t.Fatal("truncated angle accepted")
	}
}

func TestBrushOrientationMatchesQuakeAxes(t *testing.T) {
	for _, tc := range []struct{ angles, vector, want Vec3 }{
		{Vec3{0, 90, 0}, Vec3{1, 0, 0}, Vec3{0, 1, 0}},
		{Vec3{90, 0, 0}, Vec3{1, 0, 0}, Vec3{0, 0, -1}},
		{Vec3{0, 0, 90}, Vec3{0, 1, 0}, Vec3{0, 0, 1}},
	} {
		if got := rotateBrushVector(tc.vector, tc.angles); Distance(got, tc.want) > 1e-9 {
			t.Fatal(tc, got)
		}
	}
}

func TestMoverEscapeRequiresFreeEndpointAndDoesNotEnterAnotherBrush(t *testing.T) {
	planes, brush := testBoxBrush(Vec3{-8, -8, -32}, Vec3{8, 8, 32}, 0)
	g := &MapInfo{collision: &CollisionMap{planes: planes, brushes: []bspBrush{brush}, sides: []uint16{0, 1, 2, 3, 4, 5}, modelBrushes: [][]int{nil, {0}}}}
	mover := Mover{Model: 1, Angles: Vec3{0, 0, 90}}
	from, to := Vec3{20, 0, 0}, Vec3{36, 0, 0}
	if g.MoverHullClear(mover, from, to) || !g.MoverHullEscapeClear(mover, from, to) {
		t.Fatal("startsolid escape not distinguished from ordinary clear sweep")
	}
	if g.MoverHullEscapeClear(mover, from, Vec3{22, 0, 0}) || g.MoverHullEscapeClear(mover, to, from) {
		t.Fatal("allsolid or entering the mover accepted")
	}
	otherPlanes, other := testBoxBrush(Vec3{52, -8, -32}, Vec3{60, 8, 32}, 6)
	g.collision.planes = append(g.collision.planes, otherPlanes...)
	g.collision.sides = append(g.collision.sides, 6, 7, 8, 9, 10, 11)
	g.collision.brushes = append(g.collision.brushes, other)
	g.collision.modelBrushes[1] = append(g.collision.modelBrushes[1], 1)
	if g.MoverHullEscapeClear(mover, from, Vec3{90, 0, 0}) {
		t.Fatal("escape crossed another solid brush")
	}
}

func TestBSPHazardVolumesUseNativeContentsAndPlayerHull(t *testing.T) {
	planes, brush := testBoxBrush(Vec3{-40, -40, -32}, Vec3{40, 40, 0}, 0)
	brush.contents = 8
	g := &MapInfo{collision: &CollisionMap{planes: planes, brushes: []bspBrush{brush}, sides: []uint16{0, 1, 2, 3, 4, 5}, worldBrushes: []int{0}}}
	if !g.PlayerTouchesHazard(Vec3{0, 0, 16}) || g.PlayerTouchesHazard(Vec3{0, 0, 25}) {
		t.Fatal("lava hull contact not respected")
	}
	g.collision.brushes[0].contents = 16
	if !g.PlayerTouchesHazard(Vec3{50, 0, 16}) {
		t.Fatal("slime touched by hull edge not detected")
	}
	g.collision.brushes[0].contents = 32
	if g.PlayerTouchesHazard(Vec3{0, 0, 16}) {
		t.Fatal("ordinary water classified as lava/slime")
	}
}
