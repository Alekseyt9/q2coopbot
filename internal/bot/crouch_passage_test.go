package bot

import (
	"os"
	"q2coopbot/internal/quake"
	"testing"
	"time"
)

func TestBase1LowCeilingCommand(t *testing.T) {
	root := os.Getenv("Q2_SEARCH_SCAN_ROOT")
	if root == "" {
		t.Skip("requires local base1 BSP")
	}
	g, err := quake.LoadMap(root, "base1")
	if err != nil {
		t.Fatal(err)
	}
	now := time.Now()
	for _, tc := range []struct {
		name string
		x    float64
		duck bool
	}{
		{"enter", -494, true}, {"inside", -462, true}, {"exit", -400, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s := quake.Snapshot{Map: "base1", Frame: 50, Health: 100, OnGround: true, Self: quake.Vec3{tc.x, -35, -23.875}}
			goal := quake.Vec3{-250, -35, -23.875}
			p := &Planner{hasGoal: true, goalPoint: goal, World: World{Map: "base1", Geometry: &g, GeometryStatus: "ready", Navigation: "ready", Goal: "follow_teammate", Updated: now, Snapshot: s, Route: []quake.Waypoint{{Position: goal, Kind: 2}}}}
			cmd := p.commandAt(quake.UserCmd{}, now)
			if (cmd.Up < 0) != tc.duck || cmd.Forward == 0 && cmd.Side == 0 {
				t.Fatalf("%+v %+v", cmd, p.World.Command)
			}
			if tc.duck && p.World.Command.Skill != "crouch_passage" {
				t.Fatal("missing skill reason")
			}
		})
	}
}
