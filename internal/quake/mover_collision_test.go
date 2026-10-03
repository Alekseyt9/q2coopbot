package quake

import (
	"os"
	"testing"
)

func TestBase2MoverExactSupport(t *testing.T) {
	root := os.Getenv("Q2_SEARCH_SCAN_ROOT")
	if root == "" {
		t.Skip("requires BSP")
	}
	g, err := LoadMap(root, "base2")
	if err != nil {
		t.Fatal(err)
	}
	for _, z := range []float64{0, -190} {
		live := Mover{Model: 50, Origin: Vec3{0, 0, z}}
		p := Vec3{-22.125, 1408, 24.125 + z}
		d, ok := g.MoverFooting(live, p, 0.5)
		if !ok || d > 0.5 {
			t.Fatalf("moving deck missing: %v %v", d, ok)
		}
		side := p
		side[1] += 40
		if !g.MoverHullClear(live, p, side) {
			t.Fatal("safe lateral segment rejected")
		}
		side[1] += 8
		if g.MoverHullClear(live, p, side) {
			t.Fatal("platform structure ignored")
		}
		high := p
		high[2] += 50
		if _, ok := g.MoverFooting(live, high, 0.5); ok {
			t.Fatal("bounds treated as solid support")
		}
	}
	if g.MoverHullClear(Mover{Model: 99999}, Vec3{}, Vec3{}) {
		t.Fatal("unknown model trusted")
	}
}

func TestBase2RaisedBridgeIsNotHorizontalSupport(t *testing.T) {
	root := os.Getenv("Q2_SEARCH_SCAN_ROOT")
	if root == "" {
		t.Skip("requires base2 BSP")
	}
	g, err := LoadMap(root, "base2")
	if err != nil {
		t.Fatal(err)
	}
	// Observed in both native campaign seeds: model40 rolls 90 degrees
	// about its unchanged pivot. The unrotated model spans this point.
	bridge := Mover{Model: 40, Origin: Vec3{450, -958, -18}}
	at := Vec3{396, -810, 8.125}
	if _, ok := g.MoverFooting(bridge, at, 1); !ok {
		t.Fatal("fixture does not distinguish old horizontal pose")
	}
	bridge.Angles[2] = 90
	if _, ok := g.MoverFooting(bridge, at, 18); ok {
		t.Fatal("raised native bridge invented a floor")
	}
}

// The E2E side-wall fixture reuses model1 at these two translations.
// Verify actual brush collision, not just the fixture's intended bounds.
func TestBase2ElevatorSideWallFixture(t *testing.T) {
	root := os.Getenv("Q2_SEARCH_SCAN_ROOT")
	if root == "" {
		t.Skip("requires BSP")
	}
	g, err := LoadMap(root, "base2")
	if err != nil {
		t.Fatal(err)
	}
	from := Vec3{-22.125, 1408, 24.125}
	for _, y := range []float64{-4, 60} {
		wall := Mover{Model: 1, Origin: Vec3{-576, y, -24}}
		if !g.MoverHullClear(wall, from, Vec3{100, 1408, 24.125}) {
			t.Fatal("fixture blocks the central exit")
		}
		side := from
		if y < 0 {
			side[1] -= 40
		} else {
			side[1] += 40
		}
		if g.MoverHullClear(wall, from, side) {
			t.Fatal("fixture does not block lateral bypass")
		}
	}
}
