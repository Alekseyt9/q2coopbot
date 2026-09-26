package bot

import (
	"q2coopbot/internal/quake"
	"testing"
	"time"
)

func TestGapJumpKeepsControlWithoutAirborneRoute(t *testing.T) {
	p := &Planner{jump: &jumpFlight{from: quake.Vec3{}, landing: quake.Vec3{120, 0, 0}, frame: 40, speed: 200}, World: World{Map: "test", Updated: time.Now(), GeometryStatus: "ready", Navigation: "unreachable", Snapshot: quake.Snapshot{Frame: 41, Self: quake.Vec3{20, 0, 24}, Health: 100}}}
	cmd := p.commandAt(quake.UserCmd{}, p.World.Updated)
	if cmd.Forward == 0 || cmd.Up != 0 || p.World.Command.LimitReason != "jump_flight" {
		t.Fatalf("lost airborne movement: %+v %+v", cmd, p.World.Command)
	}
	p.World.Snapshot = quake.Snapshot{Frame: 47, Self: quake.Vec3{120, 0, 0}, Health: 100, OnGround: true}
	cmd = p.commandAt(quake.UserCmd{}, p.World.Updated)
	if p.jump != nil || cmd.Up != 0 || p.World.Command.LimitReason != "jump_landed" {
		t.Fatalf("landing not completed: %+v", p.World.Command)
	}
}

func TestGapJumpAbortsOnDeathTimeoutAndTeleport(t *testing.T) {
	for _, s := range []quake.Snapshot{{Frame: 42, Health: 0}, {Frame: 60, Health: 100}, {Frame: 42, Health: 100, Self: quake.Vec3{500, 0, 0}}} {
		p := &Planner{jump: &jumpFlight{frame: 40}, World: World{Snapshot: s}}
		cmd, active := p.jumpCommand(quake.UserCmd{})
		if !active || p.jump != nil || cmd.Forward != 0 || cmd.Up != 0 {
			t.Fatalf("unsafe continuation: %+v", s)
		}
	}
}
