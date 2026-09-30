package bot

import (
	"os"
	"q2coopbot/internal/quake"
	"testing"
)

func TestJail2CapturedDoorApproach(t *testing.T) {
	root := os.Getenv("Q2_SEARCH_SCAN_ROOT")
	if root == "" {
		t.Skip("assets")
	}
	g, err := quake.LoadMap(root, "jail2")
	if err != nil {
		t.Fatal(err)
	}
	self := quake.Vec3{126.625, -300.5, 8.125}
	target := quake.Vec3{106.20156860351562, -286.8999938964844, 8}
	movers := []quake.Mover{{Model: 41, Origin: quake.Vec3{0, 0, -230}}, {Model: 49, Origin: quake.Vec3{98, 0, 0}}, {Model: 50, Origin: quake.Vec3{-98, 0, 0}}}
	for _, tc := range []struct {
		step    float64
		blocked bool
	}{{40, true}, {16, false}, {8, false}} {
		model, reason := g.DoorMoveBlockStep(movers, self, target[0]-self[0], target[1]-self[1], tc.step)
		if (reason != "") != tc.blocked {
			t.Fatalf("step %g: model=%d reason=%s", tc.step, model, reason)
		}
	}
	closed := append([]quake.Mover(nil), movers...)
	closed[1].Origin = quake.Vec3{}
	closed[2].Origin = quake.Vec3{}
	if _, reason := g.DoorMoveBlockStep(closed, self, 0, 1, 40); reason != "dynamic_door_blocked" {
		t.Fatal("closed door allowed", reason)
	}
	if _, reason := g.DoorMoveBlockStep(nil, self, 0, 1, 40); reason != "dynamic_door_unobserved" {
		t.Fatal("unknown door allowed", reason)
	}
	from := quake.Vec3{57.625, -296.125, 16.125}
	p := &Planner{World: World{Geometry: &g, Goal: "follow_teammate", Route: []quake.Waypoint{{Position: quake.Vec3{68, -270.9, 16}, Kind: 2}, {Position: quake.Vec3{68, -266, 16.125}, Kind: 2}, {Position: quake.Vec3{0, -144.9, 16}, Kind: 2}}}}
	s := quake.Snapshot{Self: from, OnGround: true, Health: 68, Movers: movers}
	end, ok := p.doorRouteBypass(s)
	if !ok || end[0] != 0 || end[1] != from[1] {
		t.Fatalf("requires lateral alignment into open gap: %v %v", end, ok)
	}
	for _, mode := range []string{"closed", "unknown", "airborne", "other_goal", "jump"} {
		t.Run(mode, func(t *testing.T) {
			q := *p
			q.World = p.World
			c := s
			switch mode {
			case "closed":
				c.Movers = closed
			case "unknown":
				c.Movers = nil
			case "airborne":
				c.OnGround = false
			case "other_goal":
				q.World.Goal = "recover_health"
			case "jump":
				q.World.Route = []quake.Waypoint{{Position: quake.Vec3{0, -144.9, 16}, Kind: 11, Jump: true}}
			}
			if _, ok := q.doorRouteBypass(c); ok {
				t.Fatal("bypass must be rejected")
			}
		})
	}
}
