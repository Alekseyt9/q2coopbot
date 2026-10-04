package bot

import (
	"os"
	"q2coopbot/internal/quake"
	"testing"
)

// Reconstruct the observed command that spawned native shot 13 in the recorded
// death. Barrel positions are subsequent network observations of this fixture;
// this test is a geometry regression, not a deterministic native replay.
func TestBase1RecordedBarrelShot(t *testing.T) {
	root := os.Getenv("Q2_SEARCH_SCAN_ROOT")
	if root == "" {
		t.Skip("requires base1 BSP")
	}
	g, err := quake.LoadMap(root, "base1")
	if err != nil {
		t.Fatal(err)
	}
	s := quake.Snapshot{Health: 100, Weapon: "Blaster", Self: quake.Vec3{232.25, 5, 24.125},
		DeltaAngles: [3]int16{0, 23811, 0}, Barrels: []quake.Object{
			{ID: 306, Origin: quake.Vec3{136, 40, 0}}, {ID: 307, Origin: quake.Vec3{184, -24, 0}},
		}}
	p := Planner{World: World{Geometry: &g}}
	cmd := quake.UserCmd{Pitch: 17, Yaw: 18243, Buttons: 1, Forward: -57, Side: 57}
	got := p.guardBarrelShot(s, cmd)
	if got.Buttons&1 != 0 || got.Forward != cmd.Forward || got.Side != cmd.Side || p.World.Command.LimitReason != "barrel_blast_risk" {
		t.Fatal("recorded unsafe shot allowed or retreat changed", got, p.World.Command)
	}
}

func TestBarrelShotAndChainSafety(t *testing.T) {
	base := quake.Snapshot{Health: 100, Weapon: "Blaster", Barrels: []quake.Object{{ID: 1, Origin: quake.Vec3{150, 0, 0}}}}
	for _, tc := range []struct {
		name    string
		change  func(*quake.Snapshot)
		blocked bool
	}{
		{"direct", func(s *quake.Snapshot) {}, true},
		{"off_ray", func(s *quake.Snapshot) { s.Barrels[0].Origin[1] = 100 }, false},
		{"far", func(s *quake.Snapshot) { s.Barrels[0].Origin[0] = 700 }, false},
		{"partner", func(s *quake.Snapshot) { s.Barrels[0].Origin[0] = 700; at := quake.Vec3{720, 0, 0}; s.Teammate = &at }, true},
		{"moving_into_blast", func(s *quake.Snapshot) { s.Barrels[0].Origin[0] = 600; s.SelfVelocity[0] = 300 }, true},
		{"chain", func(s *quake.Snapshot) {
			s.Barrels[0].Origin[0] = 700
			s.Barrels = append(s.Barrels, quake.Object{Origin: quake.Vec3{450, 0, 0}}, quake.Object{Origin: quake.Vec3{200, 0, 0}})
		}, true},
		{"primed_grenade", func(s *quake.Snapshot) { s.Weapon = "Grenades" }, false},
		{"respawn", func(s *quake.Snapshot) { s.Health = 0 }, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s := base
			s.Barrels = append([]quake.Object(nil), base.Barrels...)
			tc.change(&s)
			p := Planner{}
			got := p.guardBarrelShot(s, quake.UserCmd{Buttons: 1})
			if (got.Buttons&1 == 0) != tc.blocked {
				t.Fatal(got, p.World.Command)
			}
		})
	}
}
