package bot

import (
	"q2coopbot/internal/quake"
	"testing"
	"time"
)

func TestMultiWeaponFixtureRequiresCompleteFreshStock(t *testing.T) {
	for _, fixture := range []string{"parasite_weapons", "parasite_weapons-scarce"} {
		bullets, shells := multiWeaponStock(fixture)
		s := quake.Snapshot{Weapon: "Machinegun", Ammo: int16(bullets), InventoryKnown: true, InventoryAgeFrames: 1, Inventory: []quake.InventoryItem{{Name: "Blaster", Count: 1}, {Name: "Machinegun", Count: 1}, {Name: "Shotgun", Count: 1}, {Name: "Bullets", Count: bullets}, {Name: "Shells", Count: shells}}}
		if !synchronousFixtureWeaponReady(fixture, s) {
			t.Fatal("full stock rejected")
		}
		for _, change := range []func(*quake.Snapshot){func(s *quake.Snapshot) { s.InventoryAgeFrames = 3 }, func(s *quake.Snapshot) { s.Ammo-- }, func(s *quake.Snapshot) { s.Inventory[2].Count = 0 }, func(s *quake.Snapshot) { s.Inventory[4].Count-- }} {
			bad := s
			bad.Inventory = append([]quake.InventoryItem(nil), s.Inventory...)
			change(&bad)
			if synchronousFixtureWeaponReady(fixture, bad) {
				t.Fatal("incomplete stock released")
			}
		}
	}
}

func TestSuperShotgunFixtureRequiresOwnedWeaponAndBothShells(t *testing.T) {
	s := quake.Snapshot{Weapon: "models/weapons/v_shotg2/tris.md2", Ammo: 20, InventoryKnown: true, InventoryAgeFrames: 1, Inventory: []quake.InventoryItem{{Name: "Blaster", Count: 1}, {Name: "Machinegun", Count: 1}, {Name: "Shotgun", Count: 1}, {Name: "Super Shotgun", Count: 1}, {Name: "Bullets", Count: 40}, {Name: "Shells", Count: 20}}}
	if !synchronousFixtureWeaponReady("parasite_weapons-ssg", s) {
		t.Fatal("complete SSG stock rejected")
	}
	for _, change := range []func(*quake.Snapshot){func(s *quake.Snapshot) { s.Inventory[3].Count = 0 }, func(s *quake.Snapshot) { s.Inventory[5].Count = 1 }, func(s *quake.Snapshot) { s.Weapon = "Machinegun" }, func(s *quake.Snapshot) { s.InventoryAgeFrames = 3 }} {
		bad := s
		bad.Inventory = append([]quake.InventoryItem(nil), s.Inventory...)
		change(&bad)
		if synchronousFixtureWeaponReady("parasite_weapons-ssg", bad) {
			t.Fatal("invalid SSG stock accepted")
		}
	}
}

func TestSuperShotgunBarrelGuardIncludesWaterRange(t *testing.T) {
	mate := quake.Vec3{8700, 1200, 0}
	s := quake.Snapshot{Health: 100, Weapon: "Super Shotgun", Teammate: &mate, Barrels: []quake.Object{{Origin: mate}}}
	p := Planner{}
	cmd := quake.UserCmd{Buttons: 1, Forward: 75, Side: -50, Up: 200}
	got := p.guardBarrelShot(s, cmd)
	if got.Buttons&1 != 0 || got.Forward != cmd.Forward || got.Side != cmd.Side || got.Up != cmd.Up || got.Yaw != cmd.Yaw || got.Pitch != cmd.Pitch {
		t.Fatal("SSG water barrel envelope or policy command preservation failed")
	}
}

func TestShotgunDirectOwnershipRequiresSupportedFixture(t *testing.T) {
	c := policyClient(t, "learned")
	c.testSynchronous = true
	c.planner.World.Snapshot.Weapon = "models/weapons/v_shotg/tris.md2"
	_, _, sel, direct := c.combatCommand(c.combatObservation(time.Now()), time.Now())
	if direct || sel.Fallback != "pilot_equip_not_ready" {
		t.Fatal("Shotgun escaped isolated fixture")
	}
	for _, fixture := range []string{"parasite_blaster", "parasite_machinegun", "parasite_weapons", "parasite_weapons-scarce"} {
		c.testWeaponSwitchFixture = fixture
		_, _, sel, direct = c.combatCommand(c.combatObservation(time.Now()), time.Now())
		if !direct || sel.Owner != "provider" || sel.Fallback != "" {
			t.Fatal("rules stole supported picked-up Shotgun control", fixture, sel)
		}
	}
}

func TestShotgunBarrelSpreadGuardPreservesPolicyAimAndMovement(t *testing.T) {
	s := quake.Snapshot{Health: 100, Weapon: "Shotgun", Barrels: []quake.Object{{Origin: quake.Vec3{150, 80, 0}}}}
	p := Planner{}
	cmd := quake.UserCmd{Buttons: 1, Forward: 75, Side: -50, Up: 200}
	got := p.guardBarrelShot(s, cmd)
	if got.Buttons&1 != 0 || got.Forward != cmd.Forward || got.Side != cmd.Side || got.Up != cmd.Up || got.Yaw != cmd.Yaw || got.Pitch != cmd.Pitch {
		t.Fatal("wrong pellet guard or policy action changed")
	}
}
