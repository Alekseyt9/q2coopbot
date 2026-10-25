package bot

import "q2coopbot/internal/quake"

// This is test-only equipment readiness, not a policy observation or learned
// weapon selection. The historical Shotgun teacher keeps fixed equipment;
// generated combat captures allow ordinary rules weapon choice after release.

func (c *Client) shotgunFixtureAllowsRulesWeaponChoice() bool {
	return c.testWeaponSwitchFixture != "parasite_shotgun" ||
		c.testSynchronous && c.testCombatGo && c.planner.testCombatOnly &&
			c.combatControl.mode == "rules" && !c.testTeacherVertical
}

func synchronousFixtureWeaponReady(fixture string, s quake.Snapshot) bool {
	if !s.InventoryKnown || s.InventoryAgeFrames < 0 || s.InventoryAgeFrames > 2 {
		return false
	}
	if multiWeaponFixture(fixture) {
		bullets, shells := multiWeaponStock(fixture)
		if fixture == "parasite_weapons-grenades-recovery" {
			return (s.Weapon == "Grenades" || s.Weapon == "models/weapons/v_handgr/tris.md2") && s.Ammo == 5 && inventoryCount(s, "Grenades") == 5 && inventoryCount(s, "Bullets") == bullets && inventoryCount(s, "Shells") == shells && inventoryCount(s, "Machinegun") > 0 && inventoryCount(s, "Shotgun") > 0 && inventoryCount(s, "Blaster") > 0
		}
		if fixture == "parasite_weapons-ssg" {
			return superShotgunWeapon(s.Weapon) && int(s.Ammo) == shells && inventoryCount(s, "Super Shotgun") > 0 && inventoryCount(s, "Bullets") == bullets && inventoryCount(s, "Shells") == shells && inventoryCount(s, "Machinegun") > 0 && inventoryCount(s, "Shotgun") > 0 && inventoryCount(s, "Blaster") > 0
		}
		return machinegunWeapon(s.Weapon) && int(s.Ammo) == bullets && inventoryCount(s, "Bullets") == bullets && inventoryCount(s, "Shells") == shells && inventoryCount(s, "Machinegun") > 0 && inventoryCount(s, "Shotgun") > 0 && inventoryCount(s, "Blaster") > 0
	}
	switch fixture {
	case "parasite_blaster":
		return s.Weapon == "Blaster"
	case "parasite_machinegun":
		return machinegunWeapon(s.Weapon) && s.Ammo == 100 && inventoryCount(s, "Bullets") == 100
	case "parasite_shotgun":
		return (s.Weapon == "Shotgun" || s.Weapon == "models/weapons/v_shotg/tris.md2") && s.Ammo == 20 && inventoryCount(s, "Shells") == 20
	default:
		return false
	}
}
