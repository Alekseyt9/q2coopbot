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

func TestPlannerBoardsRidesAndExitsElevator(t *testing.T) {
	gate, exit := quake.Vec3{-1, 0, -40}, quake.Vec3{1, 0, 100}
	n := &quake.Navigator{Areas: []quake.Area{{},
		{Min: quake.Vec3{-100, -10, -60}, Max: quake.Vec3{0, 10, -20}, Center: quake.Vec3{-50, 0, -40}},
		{Min: quake.Vec3{0, -10, 80}, Max: quake.Vec3{100, 10, 160}, Center: quake.Vec3{50, 0, 100}},
	}, Edges: [][]quake.Edge{{}, {{To: 2, Start: gate, End: exit, Kind: 11, Model: 1, Rise: 190, Cost: 145}}, nil}}
	goal := quake.Vec3{50, 0, 100}
	p := &Planner{Nav: n, GameClock: true, World: World{Map: "test", Geometry: &quake.MapInfo{Models: []quake.BSPModel{{}, {Min: quake.Vec3{-80, -30, -10}, Max: quake.Vec3{-20, 30, 10}}}}}}
	s := quake.Snapshot{Map: "test", Frame: 1, Self: gate, Teammate: &goal, Health: 100, OnGround: true,
		Movers: []quake.Mover{{Model: 1, Origin: quake.Vec3{0, 0, -190}}}}
	step := func(frame int, self quake.Vec3, moverZ float64) quake.UserCmd {
		s.Frame, s.Self, s.Movers[0].Origin[2] = frame, self, moverZ
		p.update(s, "")
		return p.command(quake.UserCmd{})
	}
	if cmd := step(1, gate, -190); cmd.Forward != 0 || p.World.Elevator != "wait_bottom" {
		t.Fatalf("wait at gate: cmd=%+v stage=%q", cmd, p.World.Elevator)
	}
	if cmd := step(2, gate, -190); cmd.Forward == 0 || p.World.Elevator != "board" {
		t.Fatalf("board platform: cmd=%+v stage=%q", cmd, p.World.Elevator)
	}
	center := quake.Vec3{-50, 0, -40}
	if cmd := step(3, center, -190); cmd.Forward != 0 || p.World.Elevator != "ride" {
		t.Fatalf("ride platform: cmd=%+v stage=%q", cmd, p.World.Elevator)
	}
	if cmd := step(50, quake.Vec3{-50, 0, 60}, -90); cmd.Forward != 0 || p.World.Elevator != "ride" {
		t.Fatalf("hold while moving: cmd=%+v stage=%q", cmd, p.World.Elevator)
	}
	movers := s.Movers
	s.Movers = nil
	s.Frame = 51
	p.update(s, "")
	if cmd := p.command(quake.UserCmd{}); cmd.Forward != 0 || p.World.Elevator != "ride_mover_hidden" {
		t.Fatalf("hold when mover vanishes: cmd=%+v stage=%q", cmd, p.World.Elevator)
	}
	s.Movers = movers
	if cmd := step(100, quake.Vec3{-50, 0, 150}, 0); cmd.Forward == 0 || p.World.Elevator != "exit" {
		t.Fatalf("exit at top: cmd=%+v stage=%q", cmd, p.World.Elevator)
	}
	if cmd := step(101, exit, 0); cmd.Forward != 0 || p.World.Elevator != "completed" || p.routeIndex != 2 {
		t.Fatalf("complete reach: cmd=%+v stage=%q index=%d", cmd, p.World.Elevator, p.routeIndex)
	}
}
