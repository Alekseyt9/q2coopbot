package bot

import (
	"os"
	"q2coopbot/internal/quake"
	"testing"
)

func TestBunk1RaisedPlatformExit(t *testing.T) {
	root, aas := os.Getenv("Q2_SEARCH_SCAN_ROOT"), os.Getenv("Q2_DROP_AAS")
	if root == "" || aas == "" {
		t.Skip("requires local bunk1 BSP/AAS")
	}
	g, err := quake.LoadMap(root, "bunk1")
	if err != nil {
		t.Fatal(err)
	}
	nav, err := quake.LoadAAS(aas)
	if err != nil {
		t.Fatal(err)
	}
	goal := quake.Vec3{-780, -64, 232.125}
	p := &Planner{Nav: nav, GameClock: true, World: World{Map: "bunk1", Geometry: &g}}
	s := quake.Snapshot{Map: "bunk1", Frame: 100, Health: 60, OnGround: true, Self: quake.Vec3{-969, -65.5, 232.125}, Teammate: &goal, Movers: []quake.Mover{{Model: 100}}}
	p.update(s, "")
	cmd, ok := p.platformExitCommand(quake.UserCmd{})
	if !ok || cmd.Up != 0 || cmd.Forward == 0 || p.routeKnown {
		t.Fatalf("exit not issued: %+v active=%v route=%v", cmd, ok, p.World.Route)
	}
	for _, name := range []string{"hidden_platform", "wrong_height", "airborne", "ordinary_route", "hold"} {
		t.Run(name, func(t *testing.T) {
			p.World.Snapshot = s
			p.World.Goal = "follow_teammate"
			p.World.Route = nil
			switch name {
			case "hidden_platform":
				p.World.Snapshot.Movers = nil
			case "wrong_height":
				p.World.Snapshot.Self[2] -= 64
			case "airborne":
				p.World.Snapshot.OnGround = false
			case "ordinary_route":
				p.World.Route = []quake.Waypoint{{Position: goal}}
			case "hold":
				p.World.Goal = "cover_teammate"
			}
			if _, ok := p.platformExitCommand(quake.UserCmd{}); ok {
				t.Fatal("unexpected exit")
			}
		})
	}
}
