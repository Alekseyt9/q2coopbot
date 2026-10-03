package bot

import (
	"os"
	"q2coopbot/internal/quake"
	"testing"
)

func TestBase1UpperCornerJoinsDistantRoute(t *testing.T) {
	root := os.Getenv("Q2_SEARCH_SCAN_ROOT")
	if root == "" {
		t.Skip("requires base1 BSP/AAS")
	}
	g, err := quake.LoadMap(root, "base1")
	if err != nil {
		t.Fatal(err)
	}
	n, err := quake.LoadAAS(root + "/maps/base1.aas")
	if err != nil {
		t.Fatal(err)
	}
	s := quake.Snapshot{Map: "base1", Frame: 1000, Health: 17, Gravity: 800, OnGround: true, Self: quake.Vec3{-1906.75, 1337, 120.125}, Movers: []quake.Mover{{Model: 19}}}
	route := []quake.Waypoint{
		{Kind: 2, ToArea: 1244, Position: quake.Vec3{-1648.9000244140625, 1316, 120}},
		{Kind: 2, ToArea: 1244, Position: quake.Vec3{-1644, 1316, 120.125}},
		{Kind: 2, ToArea: 1238, Position: quake.Vec3{-1616.9000244140625, 1286, 120}},
		{Kind: 2, ToArea: 1238, Position: quake.Vec3{-1612, 1286, 120.125}},
	}
	p := &Planner{Campaign: true, Nav: n, goalPoint: quake.Vec3{-1776, 1544, -55.875}, World: World{Map: s.Map, Geometry: &g, Snapshot: s, Goal: "reach_level_exit", Route: route}}
	if !p.planCornerDetour(s) {
		t.Fatal("no grounded connection to distant AAS corridor")
	}
	at := s.Self
	for _, next := range p.cornerDetour {
		if _, ok := p.cornerGroundStep(s, at, next); !ok {
			t.Fatal("unsafe detour segment", at, next)
		}
		at = next
	}
	if quake.Horizontal(at, route[0].Position) > 10 && quake.Horizontal(at, route[1].Position) > 10 {
		t.Fatal("detour did not reach route", at)
	}
	if p.jump != nil {
		t.Fatal("ground repair must not initiate a fall at low health")
	}
	// A valid corridor must not bypass the horizon or accept unsupported ground.
	if _, ok := p.cornerRouteConnection(s, s.Self, quake.Vec3{-1776, 1544, -55.875}); ok {
		t.Fatal("accepted descent as a grounded connector")
	}
}
