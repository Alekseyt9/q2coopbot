package main

import (
	"q2coopbot/internal/policy"
	"q2coopbot/internal/quake"
	"reflect"
	"testing"
)

func TestTerminalWorldTargetMasksOnlyPostmortemInventory(t *testing.T) {
	items := []quake.InventoryItem{{Name: "Machinegun", Count: -1}}
	o := policy.Observation{Health: -2, Inventory: &items}
	if _, err := worldNextFeatures(o, false, policy.WeaponFeatureVersion); err == nil {
		t.Fatal("ordinary export accepted corrupt inventory")
	}
	x, err := worldNextFeatures(o, true, policy.WeaponFeatureVersion)
	if err != nil {
		t.Fatal(err)
	}
	if len(x) != policy.WeaponFeatureWidth || x[0] != -.02 || x[policy.WeaponFeatureOffset] != 0 {
		t.Fatal("death/missingness target incorrect")
	}
	if (*o.Inventory)[0].Count != -1 {
		t.Fatal("native evidence was modified")
	}
	o.Health = 100
	if _, err = worldNextFeatures(o, true, policy.WeaponFeatureVersion); err == nil {
		t.Fatal("live inventory corruption was hidden")
	}
}

func TestContextTerminalTargetPreservesPrefixAndObservedEquipment(t *testing.T) {
	items := []quake.InventoryItem{{Name: "Railgun", Count: -1}}
	o := policy.Observation{Health: -2, Weapon: "models/weapons/v_rail/tris.md2", Inventory: &items}
	x, err := worldNextFeatures(o, true, policy.WeaponFeatureVersion)
	if err != nil {
		t.Fatal(err)
	}
	wx, err := worldNextFeatures(o, true, policy.HJEPAContextFeatureVersion)
	if err != nil || len(wx) != 871 || !reflect.DeepEqual(x, wx[:845]) || wx[845+9] != 1 {
		t.Fatal("context target changed prefix or equipment", err)
	}
	if (*o.Inventory)[0].Count != -1 {
		t.Fatal("native evidence was mutated")
	}
	o.Health = 100
	if _, err := worldNextFeatures(o, true, policy.HJEPAContextFeatureVersion); err == nil {
		t.Fatal("live corruption hidden by context export")
	}
}
