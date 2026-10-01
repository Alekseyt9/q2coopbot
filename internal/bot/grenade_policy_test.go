package bot

import (
	"os"
	"q2coopbot/internal/quake"
	"testing"
)

func TestGrenadeSampledPolicyAndRelease(t *testing.T) {
	root := os.Getenv("Q2_SEARCH_SCAN_ROOT")
	if root == "" {
		t.Skip("assets")
	}
	g, err := quake.LoadMap(root, "base1")
	if err != nil {
		t.Fatal(err)
	}
	friend := quake.Vec3{-945, 1292, 344.125}
	clear := true
	s := quake.Snapshot{Map: "base1", Frame: 140, Health: 100, Ammo: 5, Weapon: "Grenades", GunFrame: 16, OnGround: true, Gravity: 800, Self: quake.Vec3{-937, 1149, 344.125}, Teammate: &friend, TeammateEntity: 1, Enemies: []quake.Object{{ID: 90, Class: "monster_insane", Origin: quake.Vec3{-680, 1149, 296}, Solid: 8290, ClearShot: &clear}}}
	p := &Planner{World: World{Geometry: &g}, shotTeammateMotion: shotTeammateMotion{known: true, frame: 140, entity: 1}}
	plan := p.chooseGrenadeThrow(s)
	if plan == nil {
		t.Fatal("no sampled plan in isolated contact fixture")
	}
	if len(plan.Samples) != 9 {
		t.Fatal(plan)
	}
	p.World.Command = CommandDecision{AimSource: "enemy"}
	selection := s
	selection.Weapon = "Blaster"
	selection.InventoryKnown = true
	selection.Inventory = []quake.InventoryItem{{Name: "Grenades", Count: 5}}
	if p.grenadeWeaponRequest(selection, quake.UserCmd{Buttons: 1}) != "use Grenades" {
		t.Fatal("grenade selection")
	}
	selection.Weapon = "Grenades"
	selection.Frame++
	selection.GunFrame = 0
	if !p.grenadeSelectionPending(selection) {
		t.Fatal("activation switched back to Blaster")
	}
	p.World.Command = CommandDecision{AimSource: "enemy"}
	cmd := p.guardHandGrenade(s, quake.UserCmd{Buttons: 1})
	if cmd.Buttons&1 == 0 || p.World.Command.LimitReason != "hand_grenade_auto_start" || !p.grenadeThrowPending(s) {
		t.Fatal(cmd, p.World.Command)
	}
	s.Frame++
	s.Enemies = nil
	s.GunFrame = 11
	s.ViewAngles = [3]int16{plan.Pitch, plan.Yaw, 0}
	cmd = p.guardHandGrenade(s, quake.UserCmd{Buttons: 1})
	if cmd.Buttons&1 != 0 || p.World.Command.LimitReason != "hand_grenade_release" {
		t.Fatal("lost target held grenade", cmd)
	}
	// A nearby friend, even stationary, rejects the expected contact blast.
	p.grenadeThrow = nil
	s.Frame = 140
	s.GunFrame = 16
	s.Enemies = []quake.Object{{ID: 90, Class: "monster_insane", Origin: quake.Vec3{-680, 1149, 296}, Solid: 8290, ClearShot: &clear}}
	friend = quake.Vec3{-700, 1149, 344.125}
	if p.chooseGrenadeThrow(s) != nil {
		t.Fatal("friend near impact accepted")
	}
	s.Teammate = nil
	s.LastTeammate = &friend
	if p.chooseGrenadeThrow(s) != nil {
		t.Fatal("unseen friend accepted")
	}
	s.LastTeammate = nil
	p.World.Geometry = nil
	if p.chooseGrenadeThrow(s) != nil {
		t.Fatal("missing geometry accepted")
	}
}
