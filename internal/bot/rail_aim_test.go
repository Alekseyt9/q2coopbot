package bot

import (
	"q2coopbot/internal/quake"
	"testing"
	"time"
)

func TestRailAcquisitionAndPenetratingFriendlyGuard(t *testing.T) {
	clear := true
	e := quake.Object{ID: 7, Class: "monster_tank", Origin: quake.Vec3{800, 0, 0}, Solid: 8290, ClearShot: &clear}
	s := quake.Snapshot{Map: "base1", Frame: 1, Health: 100, Ammo: 10, Weapon: "models/weapons/v_rail/tris.md2", Enemies: []quake.Object{e}}
	p := &Planner{World: World{Map: "base1", GeometryStatus: "ready", Goal: "cover_teammate"}}
	prev := quake.UserCmd{Yaw: 32767}
	step := func(want bool) {
		t.Helper()
		p.World.Snapshot = s
		p.World.Updated = time.Now()
		cmd := p.commandAt(prev, p.World.Updated)
		if (cmd.Buttons&1 != 0) != want {
			t.Fatalf("frame%d cmd=%+v decision=%+v", s.Frame, cmd, p.World.Command)
		}
		prev = cmd
		s.Frame++
	}
	for i := 0; i < 4; i++ {
		step(false)
	}
	step(true)
	friend := quake.Vec3{1000, 0, 0}
	s.Teammate = &friend
	step(false)
	if p.World.Command.LimitReason != "friendly_line_of_fire" {
		t.Fatal("friend behind target ignored")
	}
	s.Teammate = nil
	step(false)
	step(true)
	s.Enemies[0].Solid = 4194
	step(false)
	step(true)
	s.Frame += 2
	step(false)
}

func TestRailAimingDoesNotStopWalking(t *testing.T) {
	clear := true
	s := quake.Snapshot{Map: "base1", Frame: 1, Health: 100, Ammo: 10, Weapon: "Railgun", Enemies: []quake.Object{{ID: 7, Origin: quake.Vec3{100, 0, 0}, ClearShot: &clear}}}
	p := &Planner{hasGoal: true, goalPoint: quake.Vec3{0, 200, 0}, World: World{Map: "base1", GeometryStatus: "ready", Navigation: "ready", Goal: "follow_teammate", Snapshot: s, Updated: time.Now()}}
	cmd := p.commandAt(quake.UserCmd{}, p.World.Updated)
	if cmd.Buttons != 0 || cmd.Forward != 0 || cmd.Side == 0 || cmd.Yaw != 0 || p.World.Command.LimitReason != "rail_aim_settling" {
		t.Fatalf("aim/move arbitration: %+v %+v", cmd, p.World.Command)
	}
}
