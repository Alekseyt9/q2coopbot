package bot

import (
	"context"
	"strings"
	"testing"

	"q2coopbot/internal/quake"
)

func TestShotgunTeacherWaitsForFreshLoadedWeapon(t *testing.T) {
	s := quake.Snapshot{Weapon: "models/weapons/v_shotg/tris.md2", Ammo: 20, InventoryKnown: true, InventoryAgeFrames: 1, Inventory: []quake.InventoryItem{{Name: "Shells", Count: 20}}}
	if !synchronousFixtureWeaponReady("parasite_shotgun", s) {
		t.Fatal("loaded Shotgun not ready")
	}
	for _, name := range []string{"wrong_weapon", "wrong_ammo", "stale_inventory", "missing_shells", "unknown_fixture"} {
		t.Run(name, func(t *testing.T) {
			bad := s
			fixture := "parasite_shotgun"
			switch name {
			case "wrong_weapon":
				bad.Weapon = "Blaster"
			case "wrong_ammo":
				bad.Ammo = 19
			case "stale_inventory":
				bad.InventoryAgeFrames = 3
			case "missing_shells":
				bad.Inventory = nil
			case "unknown_fixture":
				fixture = "other"
			}
			if synchronousFixtureWeaponReady(fixture, bad) {
				t.Fatal("invalid equipment released")
			}
		})
	}
}

func TestShotgunTeacherRejectsNormalAndLearnedUse(t *testing.T) {
	for _, mode := range []string{"rules", "learned", "learned-shadow"} {
		cfg := Config{Host: "127.0.0.1", CombatMode: mode, TestWeaponSwitchFixture: "parasite_shotgun"}
		err := Run(context.Background(), cfg)
		if err == nil || mode == "rules" && !strings.Contains(err.Error(), "fixed Shotgun teacher") || mode != "rules" && !strings.Contains(err.Error(), "direct/shadow") {
			t.Fatalf("unexpected validation %s: %v", mode, err)
		}
	}
}
