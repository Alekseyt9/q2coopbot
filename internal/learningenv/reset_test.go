package learningenv

import (
	"math"
	"q2coopbot/internal/policy"
	"q2coopbot/internal/quake"
	"testing"
)

func TestResetStockWeaponsAndSparseZeroInventory(t *testing.T) {
	o := damageStep(10).Observation
	o.Version, o.Health, o.Ammo, o.OnGround = policy.ObservationVersion, 100, 5, true
	o.Position = quake.Vec3{32, -224, 24}
	visible := true
	o.Enemies = []policy.Enemy{{Class: "monster_parasite", Relative: quake.Vec3{168, 0, 0}, ClearShot: &visible}}
	items := []quake.InventoryItem{{Name: "Blaster", Count: 1}}
	age := 0
	o.Inventory, o.InventoryAgeFrames = &items, &age
	expected := ResetExpectation{Version: ResetVersion, Map: "base1", Position: o.Position, Health: 100, Ammo: 5, EnemyClass: "monster_parasite", EnemyPosition: quake.Vec3{200, -224, 24}, Inventory: []quake.InventoryItem{{Name: "Blaster", Count: 1}, {Name: "Shells", Count: 0}}}
	for name, model := range resetWeaponModels {
		expected.Weapon = name
		o.Weapon = "models/weapons/" + model + "/tris.md2"
		if r := VerifyReset(o, expected); !r.ObservedFieldsConfirmed {
			t.Fatalf("%s: %s", name, r.Reason)
		}
	}
	items = append(items, quake.InventoryItem{Name: "Shells", Count: 1})
	if r := VerifyReset(o, expected); r.Reason != "reset_inventory_mismatch" {
		t.Fatal("unexpected ammo accepted", r)
	}
	expected.Weapon = "unknown"
	if r := VerifyReset(o, expected); r.Reason != "invalid_or_unsupported_reset_expectation" {
		t.Fatal("unknown weapon accepted", r)
	}
}

func TestGrenadesResetRequiresObservedWeaponAndAmmo(t *testing.T) {
	o := damageStep(10).Observation
	o.Version = policy.ObservationVersion
	o.Health, o.Ammo, o.OnGround = 100, 5, true
	o.Weapon = "models/weapons/v_handgr/tris.md2"
	o.Position = quake.Vec3{32, -224, 24.125}
	visible := true
	o.Enemies = []policy.Enemy{{Class: "monster_parasite", Relative: quake.Vec3{168, 0, -.125}, ClearShot: &visible}}
	expected := ResetExpectation{Version: ResetVersion, Map: "base1", Position: quake.Vec3{32, -224, 24}, Health: 100, Ammo: 5, Weapon: "Grenades", EnemyClass: "monster_parasite", EnemyPosition: quake.Vec3{200, -224, 24}}
	if r := VerifyReset(o, expected); !r.ObservedFieldsConfirmed || r.FullServerResetConfirmed {
		t.Fatal(r)
	}
	o.Weapon = "Machinegun"
	if r := VerifyReset(o, expected); r.ObservedFieldsConfirmed || r.Reason != "reset_weapon_mismatch" {
		t.Fatal("wrong current weapon accepted", r)
	}
	o.Weapon, o.Ammo = "Grenades", 4
	if r := VerifyReset(o, expected); r.ObservedFieldsConfirmed || r.Reason != "reset_resources_mismatch" {
		t.Fatal("wrong grenade count accepted", r)
	}
}

func TestPairedParticipantResetOcclusionDoesNotBypassResources(t *testing.T) {
	o := damageStep(10).Observation
	o.Version, o.Health, o.Weapon, o.OnGround = policy.ObservationVersion, 100, "Blaster", true
	o.Enemies = nil
	expected := ResetExpectation{Version: ResetVersion, Map: o.Identity.Map, Position: o.Position, Health: 100, Weapon: "Blaster", EnemyClass: "monster_soldier", AllowUnobservedEnemy: true}
	if r := VerifyReset(o, expected); r.ObservedFieldsConfirmed {
		t.Fatal("solo accepted hidden enemy opt-in", r)
	}
	r := VerifyPairedParticipantReset(o, expected)
	if !r.ObservedFieldsConfirmed || r.FullServerResetConfirmed || r.Unverified[len(r.Unverified)-1] != "enemy_presence_and_pose_not_observed" {
		t.Fatal(r)
	}
	o.Health--
	if r := VerifyPairedParticipantReset(o, expected); r.ObservedFieldsConfirmed {
		t.Fatal("occlusion hid resource mismatch", r)
	}
}

func TestResetVerifiesObservedFieldsAndKeepsHiddenStateUnconfirmed(t *testing.T) {
	s := damageStep(10)
	o := s.Observation
	o.Version = policy.ObservationVersion
	o.Health = 100
	o.Ammo = 20
	o.OnGround = true
	o.Weapon = "models/weapons/v_shotg/tris.md2"
	o.Position = quake.Vec3{32, -224, 24.125}
	visible := true
	o.Enemies = []policy.Enemy{{Class: "monster_parasite", Relative: quake.Vec3{168, 0, -.125}, ClearShot: &visible}}
	expected := ResetExpectation{Version: ResetVersion, Map: "base1", Position: quake.Vec3{32, -224, 24}, Health: 100, Ammo: 20, Weapon: "Shotgun", EnemyClass: "monster_parasite", EnemyPosition: quake.Vec3{200, -224, 24}}
	r := VerifyReset(o, expected)
	if !r.ObservedFieldsConfirmed || r.FullServerResetConfirmed || len(r.Unverified) == 0 || r.Error() != nil {
		t.Fatal(r)
	}
	for _, change := range []func(*policy.Observation){
		func(o *policy.Observation) { o.Identity.Life++ }, func(o *policy.Observation) { o.Identity.Map = "base2" },
		func(o *policy.Observation) { o.Health-- }, func(o *policy.Observation) { o.Armor++ }, func(o *policy.Observation) { o.Ammo-- },
		func(o *policy.Observation) { o.OnGround = false }, func(o *policy.Observation) { o.Ducked = true },
		func(o *policy.Observation) { o.Position[0] += 2 }, func(o *policy.Observation) { o.Position[0] = math.NaN() },
		func(o *policy.Observation) { o.Weapon = "Blaster" }, func(o *policy.Observation) { o.Enemies = nil },
		func(o *policy.Observation) { o.AgeMS = 301 },
	} {
		copy := o
		change(&copy)
		if r := VerifyReset(copy, expected); r.ObservedFieldsConfirmed || r.Error() == nil {
			t.Fatal(r)
		}
	}
	expected.Position[0] = math.NaN()
	if r := VerifyReset(o, expected); r.ObservedFieldsConfirmed {
		t.Fatal("invalid expectation accepted")
	}
}

func TestMachinegunResetRejectsWrongAmmo(t *testing.T) {
	o := damageStep(10).Observation
	o.Version = policy.ObservationVersion
	o.Health, o.Ammo, o.OnGround = 100, 100, true
	o.Weapon = "models/weapons/v_machn/tris.md2"
	o.Position = quake.Vec3{32, -224, 24.125}
	visible := true
	o.Enemies = []policy.Enemy{{Class: "monster_parasite", Relative: quake.Vec3{168, 0, -.125}, ClearShot: &visible}}
	expected := ResetExpectation{Version: ResetVersion, Map: "base1", Position: quake.Vec3{32, -224, 24}, Health: 100, Ammo: 100, Weapon: "Machinegun", EnemyClass: "monster_parasite", EnemyPosition: quake.Vec3{200, -224, 24}}
	if r := VerifyReset(o, expected); !r.ObservedFieldsConfirmed || r.FullServerResetConfirmed {
		t.Fatal(r)
	}
	o.Ammo = 99
	if r := VerifyReset(o, expected); r.ObservedFieldsConfirmed || r.Reason != "reset_resources_mismatch" {
		t.Fatal("wrong bullet count accepted", r)
	}
}

func TestResetRequiresDeclaredFreshInventory(t *testing.T) {
	o := damageStep(10).Observation
	o.Version = policy.ObservationVersion
	o.Health, o.Ammo, o.OnGround = 100, 40, true
	o.Weapon = "Machinegun"
	o.Position = quake.Vec3{32, -224, 24.125}
	visible := true
	o.Enemies = []policy.Enemy{{Class: "monster_parasite", Relative: quake.Vec3{168, 0, -.125}, ClearShot: &visible}}
	items := []quake.InventoryItem{{Name: "Shotgun", Count: 1}, {Name: "Shells", Count: 20}}
	age := 1
	o.Inventory, o.InventoryAgeFrames = &items, &age
	expected := ResetExpectation{Version: ResetVersion, Map: "base1", Position: quake.Vec3{32, -224, 24}, Health: 100, Ammo: 40, Weapon: "Machinegun", EnemyClass: "monster_parasite", EnemyPosition: quake.Vec3{200, -224, 24}, Inventory: append([]quake.InventoryItem(nil), items...)}
	if !VerifyReset(o, expected).ObservedFieldsConfirmed {
		t.Fatal("declared inventory rejected")
	}
	items[1].Count = 19
	if VerifyReset(o, expected).ObservedFieldsConfirmed {
		t.Fatal("HUD-only reset bypassed shell mismatch")
	}
	items[1].Count = 20
	age = 3
	if VerifyReset(o, expected).ObservedFieldsConfirmed {
		t.Fatal("stale inventory accepted")
	}
	age = 1
	o.Inventory = nil
	if VerifyReset(o, expected).ObservedFieldsConfirmed {
		t.Fatal("unknown inventory accepted")
	}
}
