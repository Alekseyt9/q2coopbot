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
	if _, ok := p.regroupEntryRoute(s, player); ok {
		t.Fatal("original disconnected spawn not reproduced")
	}
	p.update(s, "")
	if p.World.Goal != "regroup_after_respawn" || !p.routeOK {
		for _, max := range []float64{2, 18, 24, 32, 64} {
			drop, ok := g.GroundDrop(s.Self, max)
			t.Logf("support max=%v drop=%v ok=%v", max, drop, ok)
		}
		d := quake.Horizontal(s.Self, player)
		dx, dy := (player[0]-s.Self[0])/d, (player[1]-s.Self[1])/d
		for at := 0.0; at < d; at += 16 {
			a := quake.Vec3{s.Self[0] + dx*at, s.Self[1] + dy*at, s.Self[2]}
			b := quake.Vec3{a[0] + dx*16, a[1] + dy*16, a[2]}
			if !g.PlayerMoveClear(a, b) || g.GroundMoveHazardStep(n, a, dx, dy, 16) != "" {
				t.Logf("blocked at %v hull=%v hazard=%v", a, g.PlayerMoveClear(a, b), g.GroundMoveHazardStep(n, a, dx, dy, 16))
				break
			}
		}
		t.Fatalf("no supported return: %s %s", p.World.Goal, p.World.Navigation)
	}
	for _, bad := range []quake.Vec3{{-1000, 2520, -231.875}, {464.125, 2520, -200}, {856, 2200, -231.875}} {
		if _, ok := p.supportedReturnRoute(s, bad); ok {
			t.Fatalf("unsafe route accepted: %v", bad)
		}
	}
	s.OnGround = false
	if _, ok := p.supportedReturnRoute(s, player); ok {
		t.Fatal("airborne walk accepted")
	}
}
