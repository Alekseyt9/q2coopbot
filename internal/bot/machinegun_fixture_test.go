package bot

import (
	"testing"
	"time"

	"q2coopbot/internal/quake"
)

func TestMachinegunFixtureRequiresFreshFullAmmo(t *testing.T) {
	s := quake.Snapshot{Weapon: "models/weapons/v_machn/tris.md2", Ammo: 100, InventoryKnown: true, InventoryAgeFrames: 1, Inventory: []quake.InventoryItem{{Name: "Bullets", Count: 100}}}
	if !synchronousFixtureWeaponReady("parasite_machinegun", s) {
		t.Fatal("loaded Machinegun not ready")
	}
	for _, change := range []func(*quake.Snapshot){
		func(s *quake.Snapshot) { s.Ammo = 99 },
		func(s *quake.Snapshot) { s.InventoryAgeFrames = 3 },
		func(s *quake.Snapshot) { s.InventoryKnown = false },
		func(s *quake.Snapshot) { s.Inventory = nil },
		func(s *quake.Snapshot) { s.Weapon = "Blaster" },
	} {
		bad := s
		change(&bad)
		if synchronousFixtureWeaponReady("parasite_machinegun", bad) {
			t.Fatal("incomplete equipment released")
		}
	}
}

func TestMachinegunDirectOwnershipRequiresItsIsolatedFixture(t *testing.T) {
	c := policyClient(t, "learned")
	c.testSynchronous = true
	c.planner.World.Snapshot.Weapon = "models/weapons/v_machn/tris.md2"
	_, _, selection, direct := c.combatCommand(c.combatObservation(time.Now()), time.Now())
	if direct || selection.Fallback != "pilot_equip_not_ready" {
		t.Fatal("Machinegun enabled outside its fixture", selection)
	}
	c.testWeaponSwitchFixture = "parasite_machinegun"
	cmd, proposed, selection, direct := c.combatCommand(c.combatObservation(time.Now()), time.Now())
	if !direct || selection.Owner != "provider" || cmd.Yaw != proposed.Yaw || cmd.Pitch != proposed.Pitch {
		t.Fatal("rules replaced Machinegun policy aim", selection)
	}
}

func TestMachinegunBarrelSpreadGuardOnlyStopsFire(t *testing.T) {
	for _, tc := range []struct {
		name    string
		at      quake.Vec3
		blocked bool
	}{{"off_center_spread", quake.Vec3{150, 80, 0}, true}, {"outside_cone", quake.Vec3{150, 200, 0}, false}, {"outside_blast", quake.Vec3{700, 0, 0}, false}} {
		t.Run(tc.name, func(t *testing.T) {
			s := quake.Snapshot{Health: 100, Weapon: "Machinegun", Barrels: []quake.Object{{Origin: tc.at}}}
			p := Planner{}
			cmd := quake.UserCmd{Buttons: 1, Forward: 75, Side: -50, Up: 200}
			got := p.guardBarrelShot(s, cmd)
			if (got.Buttons&1 == 0) != tc.blocked || got.Forward != cmd.Forward || got.Side != cmd.Side || got.Up != cmd.Up || got.Yaw != cmd.Yaw || got.Pitch != cmd.Pitch {
				t.Fatal("wrong shot guard or policy movement/aim changed", got)
			}
		})
	}
}
