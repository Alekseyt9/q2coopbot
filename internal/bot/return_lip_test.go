package bot

import (
	"os"
	"q2coopbot/internal/quake"
	"testing"
)

func TestBase3ReturnLip(t *testing.T) {
	root := os.Getenv("Q2_SEARCH_SCAN_ROOT")
	if root == "" {
		t.Skip("assets")
	}
	g, e := quake.LoadMap(root, "base3")
	if e != nil {
		t.Fatal(e)
	}
	n, e := quake.LoadAAS(root + "/maps/base3.aas")
	if e != nil {
		t.Fatal(e)
	}
	s := quake.Snapshot{Map: "base3", Frame: 100, Health: 100, OnGround: true, Self: quake.Vec3{1518, 1400.25, -807.875}}
	goal := quake.Vec3{-766.25, -774.5, -279.875}
	p := &Planner{Nav: n, World: World{Map: s.Map, Geometry: &g, GeometryStatus: "ready", Snapshot: s, Goal: "regroup_after_respawn"}, goalPoint: goal}
	p.World.Route, _ = n.Route(s.Self, goal)
	if !p.planGapJump() {
		t.Fatal("no safe crossing")
	}
	if p.jump.speed >= 200 || p.jump.landing[0] >= 1440 || p.jump.landing[2] >= s.Self[2] {
		t.Fatalf("wrong landing: %+v", p.jump)
	}
	route := p.World.Route
	p.World.Route = append([]quake.Waypoint{{Kind: 11}}, route...)
	p.jump = nil
	if p.planGapJump() {
		t.Fatal("must not jump across an untraversed elevator")
	}
	p.World.Route = route
	p.World.Goal = "follow_teammate"
	if !p.planGapJump() {
		t.Fatal("same crossing must work while following")
	}
}

func TestBlockedWalkingCornerDoesNotAuthorizeJump(t *testing.T) {
	p := &Planner{World: World{Snapshot: quake.Snapshot{Self: quake.Vec3{641.75, 2504.875, -231.875}}, Route: []quake.Waypoint{{Position: quake.Vec3{640, 2521, -232}, Kind: 2}, {Position: quake.Vec3{640, 2526, -231.875}, Kind: 2}}}}
	if p.blockedDropApproach() {
		t.Fatal("ordinary walking corner must not start a gap jump")
	}
	p.World.Snapshot.Self = quake.Vec3{1518, 1400.25, -807.875}
	p.World.Route = []quake.Waypoint{{Position: quake.Vec3{1441, 1371, -780}, Kind: 7, ToArea: 490}, {Position: quake.Vec3{1439, 1371, -840}, Kind: 7, ToArea: 490}}
	if !p.blockedDropApproach() {
		t.Fatal("nearby paired drop not recognized")
	}
	p.World.Route[1].ToArea++
	if p.blockedDropApproach() {
		t.Fatal("unpaired reaches accepted")
	}
}
