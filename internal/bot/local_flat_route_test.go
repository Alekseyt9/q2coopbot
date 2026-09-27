package bot

import (
	"os"
	"path/filepath"
	"q2coopbot/internal/quake"
	"testing"
)

func TestBase1GroundReconnectRoute(t *testing.T) {
	root := os.Getenv("Q2_SEARCH_SCAN_ROOT")
	if root == "" {
		t.Skip("requires base1 BSP/AAS")
	}
	g, err := quake.LoadMap(root, "base1")
	if err != nil {
		t.Fatal(err)
	}
	n, err := quake.LoadAAS(filepath.Join(root, "maps/base1.aas"))
	if err != nil {
		t.Fatal(err)
	}
	goal := quake.Vec3{-270, -42, -23.875}
	s := quake.Snapshot{Self: quake.Vec3{-510, -42, -23.875}, OnGround: true, Teammate: &goal}
	p := &Planner{Nav: n, World: World{Goal: "follow_teammate", Geometry: &g}}
	if _, ok := n.Route(s.Self, goal); ok {
		t.Fatal("fixture no longer needs reconnect")
	}
	r, ok := p.localFlatRoute(s, goal)
	if !ok || len(r) < 2 || r[0].Position != (quake.Vec3{-510, -26, -23.875}) {
		t.Fatalf("route=%v ok=%v", r, ok)
	}
	if !g.PlayerMoveClear(s.Self, r[0].Position) || !g.CrouchStepClear(s.Self, 0, 1, 16) {
		t.Fatal("unchecked connection")
	}
	s.OnGround = false
	if _, ok := p.localFlatRoute(s, goal); ok {
		t.Fatal("airborne reconnect")
	}
	s.OnGround = true
	s.Teammate = nil
	if _, ok := p.localFlatRoute(s, goal); ok {
		t.Fatal("hidden target reconnect")
	}
	s.Teammate = &goal
	p.World.Goal = "search_last_seen"
	if _, ok := p.localFlatRoute(s, goal); ok {
		t.Fatal("expanded search policy")
	}
}
