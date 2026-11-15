package main

import (
	"q2coopbot/internal/policy"
	"q2coopbot/internal/quake"
	"testing"
)

func TestTerminalWorldTargetMasksOnlyPostmortemInventory(t *testing.T) {
	items := []quake.InventoryItem{{Name: "Machinegun", Count: -1}}
	o := policy.Observation{Health: -2, Inventory: &items}
	if _, err := worldNextFeatures(o, false); err == nil {
		t.Fatal("ordinary export accepted corrupt inventory")
	}
	x, err := worldNextFeatures(o, true)
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
	if _, err = worldNextFeatures(o, true); err == nil {
		t.Fatal("live inventory corruption was hidden")
	}
}
