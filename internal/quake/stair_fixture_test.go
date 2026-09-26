package quake

import (
	"os"
	"path/filepath"
	"testing"
)

func TestStairFixture(t *testing.T) {
	root := os.Getenv("Q2_STAIR_ROOT")
	if root == "" {
		t.Skip("set Q2_STAIR_ROOT to baseq2 assets")
	}
	nav, err := LoadAAS(filepath.Join(root, "maps", "base1.aas"))
	if err != nil {
		t.Fatal(err)
	}
	goal := Vec3{-1883.375, 1816.75, 120.125}
	for _, start := range []Vec3{{-1588.25, 1797.375, -7.875}, {-1577.5, 1788.25, -23.875}} {
		route, ok := nav.Route(start, goal)
		if !ok || len(route) == 0 {
			t.Fatalf("stairs route missing from %v", start)
		}
		for _, wp := range route {
			if wp.Kind != 2 {
				t.Fatalf("unexpected travel on stairs: %+v", wp)
			}
		}
	}
}
