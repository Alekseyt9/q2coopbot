package bot

import (
	"math"
	"os"
	"path/filepath"
	"q2coopbot/internal/quake"
	"testing"
	"time"
)

func TestBase1HealthBlockedBySoldier(t *testing.T) {
	root := os.Getenv("Q2_SEARCH_SCAN_ROOT")
	if root == "" {
		t.Skip("requires local base1 BSP/AAS")
	}
	g, err := quake.LoadMap(root, "base1")
	if err != nil {
		t.Fatal(err)
	}
	n, err := quake.LoadAAS(filepath.Join(root, "maps/base1.aas"))
	if err != nil {
		t.Fatal(err)
	}
	now := time.Now()
	visible := true
	s := quake.Snapshot{Map: "base1", Frame: 115, Health: 40, OnGround: true, Self: quake.Vec3{1122.5, 265.125, -31.875}, Weapon: "Blaster", Obstacles: []quake.Object{{ID: 316, Class: "monster_soldier", Origin: quake.Vec3{1097.375, 297.25, -31.875}, ClearShot: &visible}}}
	p := &Planner{Nav: n, hasGoal: true, healthAt: 105, goalPoint: quake.Vec3{1184, 416, -31.75}, World: World{Map: "base1", Geometry: &g, GeometryStatus: "ready", Navigation: "ready", Goal: "recover_health", Updated: now, Snapshot: s, Route: []quake.Waypoint{{Position: quake.Vec3{1136.1, 416, -32}, Kind: 2}}}}
	cmd := p.commandAt(quake.UserCmd{}, now)
	if p.World.Command.MoveSource != "resource_obstacle_avoid" || cmd.Up != 0 || cmd.Forward == 0 && cmd.Side == 0 {
		t.Fatalf("no grounded avoidance: %+v %+v", cmd, p.World.Command)
	}
	if math.Hypot(float64(cmd.Forward), float64(cmd.Side)) > 81 {
		t.Fatal("avoidance exceeded slow step speed")
	}
	ax, ay, ok := p.resourceObstacleStep(s, 14, 151)
	if !ok || g.GroundMoveHazardStep(n, s.Self, ax, ay, 16) != "" || g.DoorMoveHazard(s.Movers, s.Self, ax, ay) != "" {
		t.Fatal("unchecked avoidance")
	}
	// A second captured stall is beside the body and the corridor edge,
	// not in front of the monster. A gentle left-forward correction fits.
	edge := s
	edge.Self = quake.Vec3{1136, 303.875, -31.875}
	edge.Frame = 130
	ex, ey, escape := p.resourceObstacleStep(edge, 0.1, 112)
	if !escape || ex >= 0 || ey <= 0 {
		t.Fatalf("no corridor-edge correction: %v %v %v", ex, ey, escape)
	}
	for _, mode := range []string{"airborne", "progress", "other-goal", "no-enemy", "no-nav"} {
		t.Run(mode, func(t *testing.T) {
			q := *p
			snapshot := s
			switch mode {
			case "airborne":
				snapshot.OnGround = false
			case "progress":
				q.healthAt = s.Frame
			case "other-goal":
				q.World.Goal = "follow_teammate"
			case "behind":
				snapshot.Obstacles = []quake.Object{{Origin: quake.Vec3{1120, 230, -31.875}}}
			case "no-enemy":
				snapshot.Obstacles = nil
			case "no-nav":
				q.Nav = nil
			}
			if _, _, ok := q.resourceObstacleStep(snapshot, 14, 151); ok {
				t.Fatal("unexpected avoidance")
			}
		})
	}
}
