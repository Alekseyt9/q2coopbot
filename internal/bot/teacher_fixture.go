package bot

import "q2coopbot/internal/quake"

// This is test-only equipment readiness, not a policy observation or learned
// weapon selection. The Shotgun exercise disables automatic weapon switching.
func synchronousFixtureWeaponReady(fixture string, s quake.Snapshot) bool {
	if !s.InventoryKnown || s.InventoryAgeFrames < 0 || s.InventoryAgeFrames > 2 {
		return false
	}
	switch fixture {
	case "parasite_blaster":
		return s.Weapon == "Blaster"
	case "parasite_shotgun":
		return (s.Weapon == "Shotgun" || s.Weapon == "models/weapons/v_shotg/tris.md2") && s.Ammo == 20 && inventoryCount(s, "Shells") == 20
	default:
		return false
	}
}
