package bot

import (
	"math"
	"testing"
	"time"

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

func TestPlannerKeepsWorldRouteWhileAimingAndFiring(t *testing.T) {
	n := &quake.Navigator{Areas: []quake.Area{{},
		{Min: quake.Vec3{-10, -10, -10}, Max: quake.Vec3{10, 10, 10}},
		{Min: quake.Vec3{90, -10, -10}, Max: quake.Vec3{110, 10, 10}},
	}, Edges: [][]quake.Edge{{}, {{To: 2, Start: quake.Vec3{32, 0, 0}, End: quake.Vec3{92, 0, 0}, Kind: 2, Cost: 10}}, nil}}
	goal := quake.Vec3{100, 0, 0}
	clear := true
	p := &Planner{Nav: n, World: World{Map: "test"}}
	s := quake.Snapshot{Map: "test", Frame: 1, Self: quake.Vec3{0, 0, 0}, Teammate: &goal,
		Health: 100, Ammo: 10, Weapon: "Blaster", DeltaAngles: [3]int16{0, 8192, 0},
		Enemies: []quake.Object{{Origin: quake.Vec3{100, 100, 100}, ClearShot: &clear}}}
	p.update(s, "")
	cmd := p.command(quake.UserCmd{})
	if cmd.Buttons&1 == 0 || p.World.Command.AimSource != "enemy" || p.World.Command.MoveSource != "route" {
		t.Fatalf("combat movement arbitration: cmd=%+v decision=%+v", cmd, p.World.Command)
	}
	yaw := float64(int16(uint16(cmd.Yaw)+uint16(s.DeltaAngles[1]))) * 2 * math.Pi / 65536
	pitch := float64(int16(uint16(cmd.Pitch)+uint16(s.DeltaAngles[0]))) * 2 * math.Pi / 65536
	vx := math.Cos(pitch)*math.Cos(yaw)*float64(cmd.Forward) + math.Sin(yaw)*float64(cmd.Side)
	vy := math.Cos(pitch)*math.Sin(yaw)*float64(cmd.Forward) - math.Cos(yaw)*float64(cmd.Side)
	if vx < 200 || math.Abs(vy) > 3 {
		t.Fatalf("world movement veered from route: velocity=(%.1f,%.1f) cmd=%+v", vx, vy, cmd)
	}
}

func TestPlannerAvoidsFriendlyFireWhileFollowing(t *testing.T) {
	n := &quake.Navigator{Areas: []quake.Area{{},
		{Min: quake.Vec3{-10, -10, -10}, Max: quake.Vec3{10, 10, 10}},
		{Min: quake.Vec3{90, -10, -10}, Max: quake.Vec3{110, 10, 10}},
	}, Edges: [][]quake.Edge{{}, {{To: 2, Start: quake.Vec3{32, 0, 0}, End: quake.Vec3{92, 0, 0}, Kind: 2, Cost: 10}}, nil}}
	goal := quake.Vec3{100, 0, 0}
	clear := true
	p := &Planner{Nav: n, World: World{Map: "test"}}
	s := quake.Snapshot{Map: "test", Frame: 1, Self: quake.Vec3{0, 0, 0}, Teammate: &goal,
		Health: 100, Ammo: 10, Weapon: "Blaster", Enemies: []quake.Object{{Origin: quake.Vec3{200, 0, 0}, ClearShot: &clear}}}
	p.update(s, "")
	cmd := p.command(quake.UserCmd{})
	if cmd.Buttons != 0 || cmd.Forward <= 0 || p.World.Command.MoveSource != "route" || p.World.Command.LimitReason != "friendly_line_of_fire" {
		t.Fatalf("friendly line of fire: cmd=%+v decision=%+v", cmd, p.World.Command)
	}
}

func TestPlannerKeepsFriendlyFireReasonWithoutMovementGoal(t *testing.T) {
	goal, clear := quake.Vec3{60, 0, 0}, true
	p := &Planner{World: World{Map: "test"}}
	s := quake.Snapshot{Map: "test", Frame: 1, Self: quake.Vec3{0, 0, 0}, Teammate: &goal,
		Health: 100, Ammo: 10, Weapon: "Blaster", Enemies: []quake.Object{{Origin: quake.Vec3{120, 0, 0}, ClearShot: &clear}}}
	p.update(s, "")
	cmd := p.command(quake.UserCmd{})
	if cmd.Buttons != 0 || p.World.Command.LimitReason != "friendly_line_of_fire" {
		t.Fatalf("friendly safety reason was lost: cmd=%+v decision=%+v", cmd, p.World.Command)
	}
}

func TestTeammateBlocksShot(t *testing.T) {
	from, target := quake.Vec3{0, 0, 22}, quake.Vec3{200, 0, 22}
	for _, tc := range []struct {
		name string
		mate quake.Vec3
		want bool
	}{
		{"between", quake.Vec3{100, 0, 0}, true},
		{"side", quake.Vec3{100, 30, 0}, false},
		{"above", quake.Vec3{100, 0, 80}, false},
		{"behind", quake.Vec3{-40, 0, 0}, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := teammateBlocksShot(from, target, tc.mate); got != tc.want {
				t.Fatalf("blocked=%v, want %v", got, tc.want)
			}
		})
	}
}

func TestPlannerStopsOnStaleObservationAndRecovers(t *testing.T) {
	n := &quake.Navigator{Areas: []quake.Area{{},
		{Min: quake.Vec3{-10, -10, -10}, Max: quake.Vec3{10, 10, 10}},
		{Min: quake.Vec3{90, -10, -10}, Max: quake.Vec3{110, 10, 10}},
	}, Edges: [][]quake.Edge{{}, {{To: 2, Start: quake.Vec3{32, 0, 0}, End: quake.Vec3{92, 0, 0}, Kind: 2, Cost: 10}}, nil}}
	goal, clear := quake.Vec3{100, 0, 0}, true
	p := &Planner{Nav: n, World: World{Map: "test"}}
	s := quake.Snapshot{Map: "test", Frame: 1, Self: quake.Vec3{0, 0, 0}, Teammate: &goal,
		Health: 100, Ammo: 10, Weapon: "Blaster", Enemies: []quake.Object{{Origin: quake.Vec3{100, 100, 0}, ClearShot: &clear}}}
	p.update(s, "")
	updated := p.World.Updated
	fresh := p.commandAt(quake.UserCmd{}, updated.Add(100*time.Millisecond))
	if fresh.Forward == 0 || fresh.Buttons&1 == 0 {
		t.Fatalf("fresh command should move and fire: %+v", fresh)
	}
	stale := p.commandAt(fresh, updated.Add(301*time.Millisecond))
	if stale.Forward != 0 || stale.Side != 0 || stale.Up != 0 || stale.Buttons != 0 || p.World.Command.LimitReason != "stale_observation" {
		t.Fatalf("stale command was not neutral: cmd=%+v decision=%+v", stale, p.World.Command)
	}
	s.Frame++
	p.update(s, "")
	recovered := p.commandAt(stale, p.World.Updated.Add(100*time.Millisecond))
	if recovered.Forward == 0 || recovered.Buttons&1 == 0 || p.World.Command.LimitReason == "stale_observation" {
		t.Fatalf("fresh observation did not restore action: cmd=%+v decision=%+v", recovered, p.World.Command)
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
	if cmd := step(50, quake.Vec3{-50, 0, -25}, -175); cmd.Forward != 0 || cmd.Up >= 0 || p.World.Elevator != "ride" {
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
	if cmd := step(52, quake.Vec3{-50, 0, 0}, -150); cmd.Forward == 0 || cmd.Up >= 0 || p.World.Elevator != "exit" {
		t.Fatalf("exit during ascent: cmd=%+v stage=%q", cmd, p.World.Elevator)
	}
	if cmd := step(100, quake.Vec3{-50, 0, 150}, 0); cmd.Forward == 0 || p.World.Elevator != "exit" {
		t.Fatalf("exit at top: cmd=%+v stage=%q", cmd, p.World.Elevator)
	}
	if cmd := step(101, exit, 0); cmd.Forward != 0 || p.World.Elevator != "completed" || p.routeIndex != 2 {
		t.Fatalf("complete reach: cmd=%+v stage=%q index=%d", cmd, p.World.Elevator, p.routeIndex)
	}
}
