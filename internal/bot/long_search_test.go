package bot

import (
	"testing"

	"q2coopbot/internal/quake"
)

func TestRecentFarSightingAllowsBoundedWalkingReturn(t *testing.T) {
	last := quake.Vec3{900, 0, 24}
	age := 0
	nav := &quake.Navigator{Areas: []quake.Area{{},
		{Min: quake.Vec3{-10, -20, 0}, Max: quake.Vec3{10, 20, 48}},
		{Min: quake.Vec3{890, -20, 0}, Max: quake.Vec3{910, 20, 48}}},
		Edges: [][]quake.Edge{nil, {{To: 2, Kind: 2, Cost: 1, Start: quake.Vec3{400, 0, 24}, End: quake.Vec3{500, 0, 24}}}, nil}}
	p := &Planner{Nav: nav, World: World{Map: "test", GeometryStatus: "ready"}, GameClock: true}
	s := quake.Snapshot{Map: "test", Frame: 10, Self: quake.Vec3{0, 0, 24}, Teammate: &last,
		LastTeammate: &last, TeammateEntity: 2, LastTeammateEntity: 2,
		TeammateAgeFrames: &age, Health: 100, OnGround: true}
	p.update(s, "")
	s.Frame, s.Teammate, age = 11, nil, 1
	p.update(s, "")
	if p.World.Goal != "search_last_seen" || p.World.Navigation != "ready" || !p.longSearch ||
		p.World.SearchRoute == nil || p.World.SearchRoute.Limit != 1400 {
		t.Fatalf("recent far walking return rejected: goal=%s nav=%s check=%+v", p.World.Goal, p.World.Navigation, p.World.SearchRoute)
	}
	s.Frame, age = 210, 200
	p.update(s, "")
	if p.World.Goal != "search_last_seen" {
		t.Fatal("bounded return expired early")
	}
	s.Frame, age = 211, 201
	p.update(s, "")
	if p.World.Goal != "wait_for_teammate" || p.hasGoal {
		t.Fatal("stale far position kept as movement goal")
	}
}

func TestFarSightingStillRejectsElevatorAndUnseenOrigin(t *testing.T) {
	last := quake.Vec3{900, 0, 24}
	age := 1
	nav := &quake.Navigator{Areas: []quake.Area{{},
		{Min: quake.Vec3{-10, -20, 0}, Max: quake.Vec3{10, 20, 48}},
		{Min: quake.Vec3{890, -20, 0}, Max: quake.Vec3{910, 20, 48}}},
		Edges: [][]quake.Edge{nil, {{To: 2, Kind: 11, Model: 1, Rise: 64, Cost: 1, Start: quake.Vec3{400, 0, 24}, End: quake.Vec3{500, 0, 24}}}, nil}}
	p := &Planner{Nav: nav, World: World{Map: "test", GeometryStatus: "ready"}, GameClock: true}
	s := quake.Snapshot{Map: "test", Frame: 11, Self: quake.Vec3{0, 0, 24}, LastTeammate: &last,
		LastTeammateEntity: 2, TeammateAgeFrames: &age, Health: 100, OnGround: true}
	p.update(s, "")
	if p.World.Goal != "wait_for_teammate" {
		t.Fatal("far point without witnessed contact became goal")
	}
	p.lastSeenSelfKnown = true
	p.update(s, "")
	if p.World.Navigation != "unreachable" || p.World.SearchRoute == nil || p.World.SearchRoute.Reason != "requires_elevator" {
		t.Fatalf("unsafe far route accepted: %+v", p.World.SearchRoute)
	}
	nav.Edges[1][0].Kind = 7
	p.routeKnown = false
	p.update(s, "")
	if p.World.Navigation != "unreachable" || p.World.SearchRoute == nil || p.World.SearchRoute.Reason != "requires_nonwalking_transition" {
		t.Fatalf("ledge on far route accepted: %+v", p.World.SearchRoute)
	}
}
