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
