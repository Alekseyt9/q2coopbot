package bot

import (
	"q2coopbot/internal/quake"
	"strings"
)

func weaponModel(name string) string {
	switch name {
	case "Shotgun":
		return "/v_shotg/"
	case "Machinegun":
		return "/v_machn/"
	case "Railgun":
		return "/v_rail/"
	case "HyperBlaster":
		return "/v_hyperb/"
	}
	return ""
}

// Class is a threat estimate, never an assertion about an enemy's remaining HP.
// Explosives are deliberately not selected without splash-safety planning.
func economyWeapon(s quake.Snapshot) (string, string) {
	if !s.InventoryKnown || s.InventoryAgeFrames > 20 || s.Health <= 0 {
		return "", ""
	}
	var target *quake.Object
	level, count := 0, 0
	for i := range s.Enemies {
		e := &s.Enemies[i]
		if e.ClearShot == nil || !*e.ClearShot || quake.Distance(s.Self, e.Origin) > 650 {
			continue
		}
		threat := 0
		switch e.Class {
		case "monster_soldier":
			threat = 1
		case "monster_infantry", "monster_gunner", "monster_berserk", "monster_parasite":
			threat = 2
		case "monster_tank", "monster_supertank", "monster_gladiatr", "monster_boss2", "monster_boss3", "monster_jorg", "monster_makron":
			threat = 3
		default:
			return "", ""
		}
		count++
		if threat > level || threat == level && target != nil && quake.Distance(s.Self, e.Origin) < quake.Distance(s.Self, target.Origin) {
			target = e
			level = threat
		}
	}
	if target == nil {
		return "", ""
	}
	d := quake.Distance(s.Self, target.Origin)
	available := func(name, ammo string, reserve int) bool {
		return inventoryCount(s, name) > 0 && inventoryCount(s, ammo) >= reserve
	}
	if target.Class == "monster_parasite" {
		// Retreat beyond the tongue rather than approaching for shotgun spread.
		// A useful loaded ranged weapon stays selected while backing away.
		if s.Ammo > 0 && (strings.Contains(s.Weapon, weaponModel("Machinegun")) || strings.Contains(s.Weapon, weaponModel("HyperBlaster")) || d >= 256 && isRailgun(s.Weapon)) {
			return "", ""
		}
		if available("Machinegun", "Bullets", 1) {
			return "Machinegun", "parasite_retreat_range"
		}
		if available("HyperBlaster", "Cells", 1) {
			return "HyperBlaster", "parasite_retreat_range"
		}
		if d >= 256 && available("Railgun", "Slugs", 1) {
			return "Railgun", "parasite_retreat_range"
		}
		if inventoryCount(s, "Blaster") > 0 {
			return "Blaster", "parasite_retreat_range"
		}
		return "", ""
	}
	if s.Health < 45 {
		return "", ""
	}
	if level == 1 {
		if count != 1 || s.Health < 60 || d > 400 {
			return "", ""
		}
		// Keep inexpensive weapons already in use; don't oscillate as ranges
		// cross thresholds or switch away from an adequate Blaster.
		if s.Weapon == "Blaster" || strings.Contains(s.Weapon, "/v_shotg/") || strings.Contains(s.Weapon, "/v_machn/") {
			return "", ""
		}
		if d < 256 && available("Shotgun", "Shells", 5) {
			return "Shotgun", "save_ammo_weak_target"
		}
		if available("Machinegun", "Bullets", 21) {
			return "Machinegun", "save_ammo_weak_target"
		}
		if d <= 320 && inventoryCount(s, "Blaster") > 0 {
			return "Blaster", "save_ammo_weak_target"
		}
		return "", ""
	}
	if level == 3 && d >= 256 && available("Railgun", "Slugs", 1) {
		return "Railgun", "heavy_target"
	}
	if level == 3 && d < 256 && available("HyperBlaster", "Cells", 11) {
		return "HyperBlaster", "heavy_target"
	}
	if d <= 256 && available("Shotgun", "Shells", 1) {
		return "Shotgun", "armed_target"
	}
	if available("Machinegun", "Bullets", 1) {
		return "Machinegun", "armed_target"
	}
	return "", ""
}

func (w *weaponSwitch) economyCommand(s quake.Snapshot) string {
	name, reason := economyWeapon(s)
	if name == "" || name == "Blaster" && s.Weapon == "Blaster" || name != "Blaster" && strings.Contains(s.Weapon, weaponModel(name)) {
		w.economyCandidate = ""
		w.economyAttempts = 0
		return ""
	}
	if w.economyCandidate != name {
		w.economyCandidate = name
		w.economySince = s.Frame
		w.economyAttempts = 0
		return ""
	}
	if s.Frame-w.economySince < 2 || w.economyAttempts >= 3 || w.economyAt != 0 && s.Frame-w.economyAt < 40 {
		return ""
	}
	w.economyAt = s.Frame
	w.economyAttempts++
	w.reason = reason
	return "use " + name
}
