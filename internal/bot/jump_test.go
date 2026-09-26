package bot

import (
	"q2coopbot/internal/quake"
	"testing"
	"time"
)

func TestGapJumpKeepsControlWithoutAirborneRoute(t *testing.T) {
	p := &Planner{jump: &jumpFlight{from: quake.Vec3{}, landing: quake.Vec3{120, 0, 0}, frame: 40, speed: 200, phase: 2}, World: World{Map: "test", Updated: time.Now(), GeometryStatus: "ready", Navigation: "unreachable", Snapshot: quake.Snapshot{Frame: 41, Self: quake.Vec3{20, 0, 24}, Health: 100}}}
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
	for _, s := range []quake.Snapshot{{Frame: 42, Health: 0}, {Frame: 80, Health: 100}, {Frame: 42, Health: 100, Self: quake.Vec3{500, 0, 0}}} {
		p := &Planner{jump: &jumpFlight{frame: 40}, World: World{Snapshot: s}}
		cmd, active := p.jumpCommand(quake.UserCmd{})
		if !active || p.jump != nil || cmd.Forward != 0 || cmd.Up != 0 {
			t.Fatalf("unsafe continuation: %+v", s)
		}
	}
}

func TestGapJumpRunupPrecedesTakeoff(t *testing.T) {
	p := &Planner{jump: &jumpFlight{from: quake.Vec3{}, runup: quake.Vec3{-48, 0, 0}, landing: quake.Vec3{150, 0, 0}, frame: 40, speed: 280}, World: World{Snapshot: quake.Snapshot{Frame: 40, Health: 100, OnGround: true}}}
	cmd, _ := p.jumpCommand(quake.UserCmd{})
	if cmd.Up != 0 || p.World.Command.LimitReason != "jump_prepare" {
		t.Fatal("jumped without runup")
	}
	p.World.Snapshot.Self = quake.Vec3{-48, 0, 0}
	p.World.Snapshot.Frame++
	cmd, _ = p.jumpCommand(quake.UserCmd{})
	if cmd.Up != 0 || p.World.Command.LimitReason != "jump_runup" {
		t.Fatal("runup phase missing")
	}
	p.World.Snapshot.Self = quake.Vec3{-8, 0, 0}
	p.World.Snapshot.Frame++
	cmd, _ = p.jumpCommand(quake.UserCmd{})
	if cmd.Up == 0 || p.World.Command.LimitReason != "jump_takeoff" {
		t.Fatal("takeoff missing")
	}
	p.World.Snapshot.Frame++
	p.World.Snapshot.OnGround = false
	cmd, _ = p.jumpCommand(quake.UserCmd{})
	if cmd.Up != 0 || cmd.Forward == 0 {
		t.Fatal("did not release jump and maintain flight")
	}
}

func TestGapJumpUnexpectedFallDuringRunupCancels(t *testing.T) {
	p := &Planner{jump: &jumpFlight{frame: 40}, World: World{Snapshot: quake.Snapshot{Frame: 41, Health: 100}}}
	cmd, _ := p.jumpCommand(quake.UserCmd{})
	if p.jump != nil || cmd.Up != 0 || p.World.Command.LimitReason != "runup_lost_ground" {
		t.Fatal("unexpected fall was treated as planned jump")
	}
}
