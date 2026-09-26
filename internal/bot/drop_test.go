package bot

import (
	"os"
	"q2coopbot/internal/quake"
	"testing"
)

func TestBunk1WalkOff(t *testing.T) {
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
	goal := quake.Vec3{519, -932.5, -103.875}
	s := quake.Snapshot{Map: "bunk1", Frame: 100, Health: 98, OnGround: true, Self: quake.Vec3{463.25, -1109, 24.125}, Teammate: &goal}
	p := &Planner{Nav: nav, GameClock: true, World: World{Map: "bunk1", Geometry: &g}}
	p.update(s, "")
	if !p.planWalkOff() {
		t.Fatalf("no walk off: %v", p.World.Route)
	}
	for _, name := range []string{"no_reach", "deep_drop", "hazardous_floor"} {
		t.Run(name, func(t *testing.T) {
			p.jump = nil
			original := append([]quake.Waypoint(nil), p.World.Route...)
			defer func() { p.World.Route = original }()
			switch name {
			case "no_reach":
				p.World.Route = nil
			case "deep_drop":
				p.World.Route[1].Position[2] = -400
			case "hazardous_floor":
				areas := append([]quake.Area(nil), nav.Areas...)
				defer func() { nav.Areas = areas }()
				for i := range nav.Areas {
					nav.Areas[i].Contents |= 2
				}
			}
			if p.planWalkOff() {
				t.Fatal("unsafe drop accepted")
			}
		})
	}
}

func TestWalkOffBrakesAndNeverJumps(t *testing.T) {
	p := &Planner{jump: &jumpFlight{from: quake.Vec3{0, 0, 128}, landing: quake.Vec3{40, 0, 0}, frame: 10, phase: 2, speed: 80, drop: true}, World: World{Snapshot: quake.Snapshot{Self: quake.Vec3{0, 0, 128}, Frame: 10, Health: 100, OnGround: true}}}
	for frame := 10; frame <= 13; frame++ {
		p.World.Snapshot.Frame = frame
		if frame == 13 {
			p.World.Snapshot.OnGround = false
		}
		cmd, _ := p.jumpCommand(quake.UserCmd{})
		if cmd.Up != 0 || frame < 12 && (cmd.Forward != 0 || cmd.Side != 0) || frame >= 12 && cmd.Forward == 0 {
			t.Fatalf("frame %d: %+v", frame, cmd)
		}
	}
	p.World.Snapshot = quake.Snapshot{Frame: 18, Health: 100, Self: quake.Vec3{40, 0, 0}, OnGround: true}
	p.jumpCommand(quake.UserCmd{})
	if p.jump != nil || p.World.Command.LimitReason != "drop_landed" {
		t.Fatal("landing not confirmed")
	}
}
