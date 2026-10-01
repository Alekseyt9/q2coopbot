package bot

import (
	"q2coopbot/internal/quake"
	"testing"
	"time"
)

func TestHandGrenadeGuardPreservesOtherControls(t *testing.T) {
	p := &Planner{}
	cmd := quake.UserCmd{Buttons: 3, Forward: 200, Pitch: 17}
	for _, weapon := range []string{"Grenades", "models/weapons/v_handgr/tris.md2"} {
		s := quake.Snapshot{Health: 100, Weapon: weapon}
		got := p.guardHandGrenade(s, cmd)
		if got.Buttons != 2 || got.Forward != 200 || got.Pitch != 17 {
			t.Fatal(got)
		}
		s.Health = 0
		if p.guardHandGrenade(s, cmd).Buttons != 3 {
			t.Fatal("respawn attack suppressed")
		}
	}
	if p.guardHandGrenade(quake.Snapshot{Health: 100, Weapon: "models/weapons/v_launch/tris.md2"}, cmd) != cmd {
		t.Fatal("launcher mistaken for hand grenade")
	}
}

func TestHandGrenadeCannotUseGenericEnemyAttack(t *testing.T) {
	now := time.Now()
	clear := true
	p := &Planner{World: World{Map: "base1", Updated: now, GeometryStatus: "ready", Snapshot: quake.Snapshot{Map: "base1", Frame: 10, Health: 100, Ammo: 5, Weapon: "models/weapons/v_handgr/tris.md2", Enemies: []quake.Object{{ID: 9, Origin: quake.Vec3{200, 0, 0}, ClearShot: &clear}}}}}
	if cmd := p.commandAt(quake.UserCmd{}, now); cmd.Buttons&1 != 0 || p.World.Command.LimitReason != "hand_grenade_requires_safe_throw" {
		t.Fatal(cmd, p.World.Command)
	}
}

func TestHandGrenadeSwitchEvenWithLowHealthOrUnknownInventory(t *testing.T) {
	w := &weaponSwitch{}
	s := quake.Snapshot{Map: "base1", Frame: 10, Health: 12, Ammo: 5, Weapon: "models/weapons/v_handgr/tris.md2"}
	if got := w.command(s); got != "use Blaster" {
		t.Fatal(got)
	}
	s.Frame++
	if w.command(s) != "" {
		t.Fatal("request flooded")
	}
	s.Frame = 30
	if w.command(s) != "use Blaster" {
		t.Fatal("retry missing")
	}
}

func TestArmedGrenadeReleasesWithoutTargetAndKeepsServerPose(t *testing.T) {
	for frame := 1; frame <= 15; frame++ {
		p := &Planner{}
		s := quake.Snapshot{Health: 100, Weapon: "Grenades", GunFrame: frame, ViewAngles: [3]int16{12, -30, 7}, DeltaAngles: [3]int16{2, -10, 1}}
		cmd := p.guardHandGrenade(s, quake.UserCmd{Buttons: 3, Forward: 80, Yaw: 300, Pitch: 40})
		if cmd.Buttons != 2 || cmd.Pitch != 10 || cmd.Yaw != -20 || cmd.Roll != 6 || cmd.Forward != 80 {
			t.Fatalf("gunframe%d: %+v", frame, cmd)
		}
		w := &weaponSwitch{}
		s.Map = "base1"
		s.Frame = 100
		if w.command(s) != "" {
			t.Fatal("weapon changed in native firing cycle")
		}
		if frame == 11 && p.World.Command.LimitReason != "hand_grenade_release" {
			t.Fatal("release not recorded")
		}
	}
}

func TestGrenadePosePreservesCheckedMovementDirection(t *testing.T) {
	p := &Planner{}
	s := quake.Snapshot{Health: 100, Weapon: "Grenades", GunFrame: 11}
	cmd := p.guardHandGrenade(s, quake.UserCmd{Yaw: 16384, Forward: 100, Buttons: 1})
	if cmd.Yaw != 0 || cmd.Forward != 0 || cmd.Side != -100 || cmd.Buttons != 0 {
		t.Fatalf("checked northward movement changed: %+v", cmd)
	}
}
