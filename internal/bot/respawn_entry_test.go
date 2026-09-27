package bot

import (
	"os"
	"q2coopbot/internal/quake"
	"testing"
)

func TestBase2RespawnEntry(t *testing.T) {
	root := os.Getenv("Q2_SEARCH_SCAN_ROOT")
	if root == "" {
		t.Skip("assets")
	}
	g, e := quake.LoadMap(root, "base2")
	if e != nil {
		t.Fatal(e)
	}
	n, e := quake.LoadAAS(root + "/maps/base2.aas")
	if e != nil {
		t.Fatal(e)
	}
	p := &Planner{Nav: n, World: World{Geometry: &g}}
	r, ok := p.regroupEntryRoute(quake.Snapshot{Self: quake.Vec3{828, 2232, -231.875}, OnGround: true}, quake.Vec3{194, 1940, -167.875})
	if !ok {
		t.Fatal("spawn entry remains unreachable")
	}
	t.Logf("route=%v", r)
}
