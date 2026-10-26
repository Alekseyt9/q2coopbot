package main

import (
	"encoding/json"
	"q2coopbot/internal/learningenv"
	"q2coopbot/internal/policy"
	"q2coopbot/internal/quake"
	"strings"
	"testing"
)

func TestCUDATerminalBootstrapDoesNotEncodeDeadInventory(t *testing.T) {
	items := []quake.InventoryItem{{Name: "Grenades", Count: -1}}
	next := policy.Observation{Health: -5, Inventory: &items}
	next.Identity.Frame = 142
	s := learningenv.Step{Next: &next, Terminal: true}
	s.Observation.Identity.Frame = 141
	var exported row
	if err := cudaNextFeatures(&exported, &s, policy.WeaponFeatureVersion, true); err != nil {
		t.Fatal(err)
	}
	if !exported.BootstrapZero || exported.NextFeatures != nil {
		t.Fatal("Terminal state must not be encoded for the critic")
	}
	b, _ := json.Marshal(exported)
	if strings.Contains(string(b), "next_features") {
		t.Fatal("Omitted terminal features must remain omitted on disk")
	}
	// A living/nonzero bootstrap must still reject invalid inventory. No
	// clamping, invented ammo, or relaxed feature admission is introduced.
	next.Health, s.Terminal = 80, false
	exported = row{}
	if err := cudaNextFeatures(&exported, &s, policy.WeaponFeatureVersion, false); err == nil {
		t.Fatal("Invalid nonterminal inventory accepted")
	}
	items[0].Count = 1
	if err := cudaNextFeatures(&exported, &s, policy.WeaponFeatureVersion, false); err != nil {
		t.Fatal(err)
	}
	if exported.BootstrapZero || len(exported.NextFeatures) != policy.WeaponFeatureWidth {
		t.Fatal("Living next state was not encoded")
	}
}
