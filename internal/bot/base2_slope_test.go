package bot

import (
	"os"
	"q2coopbot/internal/quake"
	"testing"
)

func TestBase2SlopeCornerRecovery(t *testing.T) {
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
	s := quake.Snapshot{Map: "base2", Frame: 177, Health: 100, OnGround: true, Self: quake.Vec3{85.625, 2225.125, -169.75}}
	goal := quake.Vec3{194, 1940, -167.875}
	p := &Planner{Nav: n, World: World{Geometry: &g, Snapshot: s, Goal: "regroup_after_respawn"}, goalPoint: goal}
	p.World.Route, _ = n.Route(s.Self, goal)
	target := p.World.Route[0].Position
	if g.GroundMoveHazardStep(n, s.Self, target[0]-s.Self[0], target[1]-s.Self[1], 16) != "static_hull_blocked" {
		t.Fatal("original obstruction not reproduced")
	}
	x, y, ok := p.regroupCornerStep(s, target)
	if !ok || x != 0 || y != 16 {
		t.Fatalf("expected checked northward recovery, got %v %v %v", x, y, ok)
	}
	if g.GroundMoveHazardStep(n, s.Self, x, y, 16) != "" {
		t.Fatal("recovery enters an unsupported or blocked landing")
	}
	s.OnGround = false
	if _, _, ok := p.regroupCornerStep(s, target); ok {
		t.Fatal("airborne snapshot must not authorize a ground step")
	}
	s.OnGround = true
	p.World.Route[0].ToArea = 404
	if _, _, ok := p.regroupCornerStep(s, target); ok {
		t.Fatal("other reaches must keep the ordinary distance limit")
	}
	p.World.Route[0].ToArea = 405
	p.World.Goal = "follow_teammate"
	if _, _, ok := p.regroupCornerStep(s, target); ok {
		t.Fatal("recovery scope unexpectedly widened")
	}
}
