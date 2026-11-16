package bot

import (
	"encoding/json"
	"q2coopbot/internal/policy"
	"q2coopbot/internal/quake"
	"testing"
)

func TestWeaponTrainingAdviceObservedOnlyAndGrenadeCycle(t *testing.T) {
	age := 0
	items := []quake.InventoryItem{{Name: "Blaster", Count: 1}, {Name: "Grenades", Count: 5}}
	o := policy.Observation{Health: 80, Weapon: "models/weapons/v_handgr/tris.md2", Ammo: 5, Inventory: &items, InventoryAgeFrames: &age}
	before, _ := json.Marshal(o)
	name, reason, err := WeaponTrainingAdvice(o)
	if err != nil || name != "Blaster" || reason != "hand_grenade_requires_safe_throw" {
		t.Fatalf("%q %q %v", name, reason, err)
	}
	after, _ := json.Marshal(o)
	if string(before) != string(after) {
		t.Fatal("teacher mutated observation")
	}
	o.GunFrame = 10
	name, reason, err = WeaponTrainingAdvice(o)
	if err != nil || name != "" || reason != "hand_grenade_wait_release" {
		t.Fatalf("armed grenade advice: %q %q %v", name, reason, err)
	}
	age = 21
	_, reason, err = WeaponTrainingAdvice(o)
	if err != nil || reason != "" {
		t.Fatal("stale inventory generated advice")
	}
	o.Inventory = nil
	_, reason, err = WeaponTrainingAdvice(o)
	if err != nil || reason != "" {
		t.Fatal("unknown inventory generated advice")
	}
}

func TestWeaponTrainingAdviceMasksEmptyWeaponPreference(t *testing.T) {
	age := 0
	items := []quake.InventoryItem{{Name: "Blaster", Count: 1}, {Name: "Machinegun", Count: 1}, {Name: "Bullets", Count: 2}}
	o := policy.Observation{Health: 80, Weapon: "models/weapons/v_rail/tris.md2", Ammo: 0, Inventory: &items, InventoryAgeFrames: &age}
	name, reason, err := WeaponTrainingAdvice(o)
	if err != nil || name != "Machinegun" || reason != "empty_weapon" {
		t.Fatalf("%q %q %v", name, reason, err)
	}
}
