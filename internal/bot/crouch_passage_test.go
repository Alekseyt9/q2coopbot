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

func TestBase1DirectCrouchRoute(t *testing.T) {
	root := os.Getenv("Q2_SEARCH_SCAN_ROOT")
	if root == "" {
		t.Skip("requires local base1 BSP")
	}
	g, err := quake.LoadMap(root, "base1")
	if err != nil {
		t.Fatal(err)
	}
	goal := quake.Vec3{-270, -35, -23.875}
	s := quake.Snapshot{Self: quake.Vec3{-510, -35, -23.875}, Teammate: &goal, OnGround: true}
	p := &Planner{World: World{Goal: "follow_teammate", Geometry: &g}}
	route, ok := p.directCrouchRoute(s, goal)
	if !ok || len(route) != 1 || route[0].Position != goal {
		t.Fatal("verified low passage rejected")
	}
	// Seven units farther south the passage cannot fit even a ducked hull.
	s.Self[1], goal[1] = -42, -42
	if g.CrouchMoveClear(quake.Vec3{-462, -42, -23.875}, quake.Vec3{-462, -42, -23.875}) {
		t.Fatal("negative fixture is not blocked")
	}
	if _, ok = p.directCrouchRoute(s, goal); ok {
		t.Fatal("too-tight shortcut accepted")
	}
	s.Self[1], goal[1] = -35, -35
	s.OnGround = false
	if _, ok = p.directCrouchRoute(s, goal); ok {
		t.Fatal("airborne shortcut accepted")
	}
	s.OnGround = true
	s.Teammate = nil
	if _, ok = p.directCrouchRoute(s, goal); ok {
		t.Fatal("unobserved teammate accepted")
	}
	s.Teammate = &goal
	p.World.Goal = "recover_health"
	if _, ok = p.directCrouchRoute(s, goal); ok {
		t.Fatal("shortcut changed health objective")
	}
	p.World.Goal = "follow_teammate"
	goal[2] += 16
	if _, ok = p.directCrouchRoute(s, goal); ok {
		t.Fatal("height change accepted")
	}
}
