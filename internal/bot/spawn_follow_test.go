package bot

import (
	"os"
	"q2coopbot/internal/quake"
	"testing"
)

func TestBase2SpawnFollowsVisiblePlayerThroughGraphEntry(t *testing.T) {
	root := os.Getenv("Q2_SEARCH_SCAN_ROOT")
	if root == "" {
		t.Skip("assets")
	}
	g, err := quake.LoadMap(root, "base2")
	if err != nil {
		t.Fatal(err)
	}
	n, err := quake.LoadAAS(root + "/maps/base2.aas")
	if err != nil {
		t.Fatal(err)
	}
	player := quake.Vec3{600, 2520, -231.875}
	s := quake.Snapshot{Map: "base2", Frame: 145, Health: 100, OnGround: true, Self: quake.Vec3{876, 2232, -231.875}, Teammate: &player}
	if _, ok := n.Route(s.Self, player); ok {
		t.Fatal("disconnected spawn no longer reproduced")
	}
	p := &Planner{Nav: n, World: World{Map: "base2", Geometry: &g, GeometryStatus: "ready"}}
	p.update(s, "")
	if p.World.Goal != "follow_teammate" || p.World.Navigation != "ready" || !p.routeOK || len(p.route) == 0 {
		t.Fatalf("visible player cannot be followed: goal=%s nav=%s", p.World.Goal, p.World.Navigation)
	}
	s.OnGround = false
	if _, ok := p.regroupEntryRoute(s, player); ok {
		t.Fatal("airborne spawn authorized ground entry")
	}
}

func TestBase2SecondSpawnReturnsToRememberedPlayer(t *testing.T) {
	root := os.Getenv("Q2_SEARCH_SCAN_ROOT")
	if root == "" {
		t.Skip("assets")
	}
	g, err := quake.LoadMap(root, "base2")
	if err != nil {
		t.Fatal(err)
	}
	n, err := quake.LoadAAS(root + "/maps/base2.aas")
	if err != nil {
		t.Fatal(err)
	}
	player := quake.Vec3{464.125, 2520, -231.875}
	s := quake.Snapshot{Map: "base2", Frame: 355, Health: 100, OnGround: true, Self: quake.Vec3{856, 2356, -231.875}, LastTeammate: &player, LastTeammateEntity: 2}
	p := &Planner{Nav: n, World: World{Map: "base2", Geometry: &g, GeometryStatus: "ready"}, respawnRegroup: &respawnRegroup{entity: 2, target: player}}
	if _, ok := n.Route(s.Self, player); ok {
		t.Fatal("original disconnected spawn not reproduced")
	}
	s.Movers = []quake.Mover{{ID: 293, Model: 44}, {ID: 290, Model: 41}, {ID: 291, Model: 42}, {ID: 292, Model: 43}}
	if _, ok := p.regroupEntryRouteStep(s, player, 16); ok {
		t.Fatal("coarse grid no longer reproduces the missed entry")
	}
	p.update(s, "")
	if p.World.Goal != "regroup_after_respawn" || !p.routeOK {
		t.Fatalf("no supported return: %s %s", p.World.Goal, p.World.Navigation)
	}
	s.OnGround = false
	if _, ok := p.regroupEntryRoute(s, player); ok {
		t.Fatal("airborne walk accepted")
	}
}
