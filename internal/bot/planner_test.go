package bot

import (
	"testing"

	"q2coopbot/internal/quake"
)

func TestPlannerKeepsWaypointProgress(t *testing.T) {
	n := &quake.Navigator{Areas: []quake.Area{{}, {Min: quake.Vec3{-10, -10, -10}, Max: quake.Vec3{10, 10, 10}}, {Min: quake.Vec3{90, -10, -10}, Max: quake.Vec3{110, 10, 10}}}, Edges: [][]quake.Edge{{}, {{To: 2, Start: quake.Vec3{32, 0, 0}, End: quake.Vec3{92, 0, 0}, Kind: 2, Cost: 10}}, nil}}
	goal := quake.Vec3{100, 0, 0}
	p := &Planner{Nav: n, World: World{Map: "test"}}
	s := quake.Snapshot{Map: "test", Frame: 1, Self: quake.Vec3{0, 0, 0}, Teammate: &goal, Health: 100}
	p.update(s, "")
	if len(p.World.Route) != 2 {
		t.Fatalf("initial route=%v", p.World.Route)
	}
	s.Self = quake.Vec3{32, 0, 0}
	s.Frame = 2
	p.update(s, "")
	if len(p.World.Route) != 1 || p.World.Route[0].Position != (quake.Vec3{92, 0, 0}) {
		t.Fatalf("route progress lost: %+v", p.World.Route)
	}
}
