package bot

import (
	"os"
	"q2coopbot/internal/quake"
	"testing"
)

func TestGrenadeClearanceCurrentClasses(t *testing.T) {
	s := quake.Snapshot{Projectiles: []quake.Object{{Class: "grenade", Origin: quake.Vec3{30, 0, 0}}, {Class: "hand_grenade", Origin: quake.Vec3{50, 0, 0}}, {Class: "unknown", Origin: quake.Vec3{}}}}
	if grenadeClearance(s, quake.Vec3{}) != 30 {
		t.Fatal("unknown projectile changed grenade distance")
	}
	s.Projectiles = nil
	if grenadeClearance(s, quake.Vec3{}) < 160 {
		t.Fatal("missing grenade retained")
	}
}

func TestBase1GrenadeEvade(t *testing.T) {
	root := os.Getenv("Q2_SEARCH_SCAN_ROOT")
	if root == "" {
		t.Skip("requires base1 BSP")
	}
	g, err := quake.LoadMap(root, "base1")
	if err != nil {
		t.Fatal(err)
	}
	s := quake.Snapshot{Health: 100, OnGround: true, Self: quake.Vec3{32, -224, 24.125}, Projectiles: []quake.Object{{ID: 900, Class: "grenade", Origin: quake.Vec3{80, -224, 24}}}}
	p := &Planner{World: World{Geometry: &g, Snapshot: s, Command: CommandDecision{AimSource: "enemy"}}}
	cmd, ok := p.combatGrenadeEvade(quake.UserCmd{Buttons: 1}, nil)
	to := p.World.Command.MovePoint
	if !ok || to == nil || cmd.Buttons != 1 || cmd.Forward == 0 && cmd.Side == 0 || p.World.Command.MoveSource != "combat_grenade_evade" {
		t.Fatal("grenade escape absent or fire discarded", cmd, p.World.Command)
	}
	if grenadeClearance(s, *to) <= grenadeClearance(s, s.Self)+8 || !coverWalkClear(p.World, s.Self, *to) {
		t.Fatal("unsafe or ineffective edge", to)
	}
	for _, change := range []func(*quake.Snapshot){
		func(s *quake.Snapshot) { s.Health = 0 },
		func(s *quake.Snapshot) { s.OnGround = false },
		func(s *quake.Snapshot) { s.Projectiles = nil },
		func(s *quake.Snapshot) { s.Projectiles[0].Origin = quake.Vec3{1000, 1000, 24} },
		func(s *quake.Snapshot) { s.Teammate = &s.Self },
	} {
		copyS := s
		copyS.Projectiles = append([]quake.Object(nil), s.Projectiles...)
		change(&copyS)
		p.World.Snapshot = copyS
		if _, ok := p.combatGrenadeEvade(quake.UserCmd{}, nil); ok {
			t.Fatal("ineligible state evaded", copyS)
		}
	}
	p.World.Snapshot = s
	p.testSetupHold = true
	if _, ok := p.combatGrenadeEvade(quake.UserCmd{}, nil); ok {
		t.Fatal("fixture setup moved")
	}
}
