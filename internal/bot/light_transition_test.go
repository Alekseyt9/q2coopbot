package bot

import (
	"os"
	"q2coopbot/internal/quake"
	"testing"
)

func TestMapTransitionReplacesLightingAndClearsMissingMap(t *testing.T) {
	root := os.Getenv("Q2_SEARCH_SCAN_ROOT")
	if root == "" {
		t.Skip("BSP assets required")
	}
	p := &Planner{}
	p.setMap("base1", root)
	old := p.World.Geometry
	if _, ok := old.StaticLightLevel(quake.Vec3{608, 192, -9.875}, nil, 1); !ok {
		t.Fatal("base1 light missing")
	}
	p.setMap("base2", root)
	if p.World.Geometry == nil || p.World.Geometry == old || p.World.Geometry.Name != "base2" {
		t.Fatal("old map geometry retained")
	}
	if p.World.Geometry.LightingError != "" {
		t.Fatal(p.World.Geometry.LightingError)
	}
	point := quake.Vec3{848, 2292, -192}
	if got, ok := p.World.Geometry.StaticLightLevel(point, nil, 1); !ok || got != 18 {
		t.Fatalf("base2 entry lighting=%d known=%v", got, ok)
	}
	if got, _ := old.StaticLightLevel(point, nil, 1); got == 18 {
		t.Fatal("fixture does not distinguish stale base1 lighting")
	}
	p.setMap("missing_light_test_map", root)
	if _, ok := p.World.Geometry.StaticLightLevel(quake.Vec3{608, 192, -9.875}, nil, 1); ok {
		t.Fatal("missing map reused previous lighting")
	}
}
