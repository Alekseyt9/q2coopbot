package bot

import (
	"q2coopbot/internal/quake"
	"strings"
)

// Server frames bound retries independently of wall-clock acceleration.
// weapnext delegates inventory/ammo validation to the authoritative server.
type weaponSwitch struct {
	parasiteSeenAt                           int
	economyCandidate, reason                 string
	economySince, economyAt, economyAttempts int
	mapName                                  string
	lastFrame, emptyAt, requestAt, attempts  int
}

func (w *weaponSwitch) command(s quake.Snapshot) string {
	w.reason = ""
	if w.mapName != s.Map || s.Frame < w.lastFrame || s.Health <= 0 {
		*w = weaponSwitch{mapName: s.Map}
	}
	if s.Frame <= w.lastFrame {
		return ""
	}
	w.lastFrame = s.Frame
	if s.Health <= 0 || s.Map == "" || s.Weapon == "" {
		return ""
	}
	for _, e := range s.Enemies {
		if e.Class == "monster_parasite" && e.ClearShot != nil && *e.ClearShot && quake.Distance(s.Self, e.Origin) <= 352 {
			w.parasiteSeenAt = s.Frame
		}
	}
	if isHandGrenade(s.Weapon) {
		if s.GunFrame >= 1 && s.GunFrame <= 15 {
			w.reason = "hand_grenade_wait_release"
			return ""
		}
		if w.requestAt != 0 && s.Frame-w.requestAt < 20 {
			return ""
		}
		w.requestAt = s.Frame
		w.reason = "hand_grenade_requires_safe_throw"
		return "use Blaster"
	}
	if s.Weapon == "Blaster" || s.Ammo > 0 {
		w.emptyAt, w.requestAt, w.attempts = 0, 0, 0
		return w.economyCommand(s)
	}
	if w.emptyAt == 0 {
		w.emptyAt = s.Frame
	}
	if s.Frame-w.emptyAt < 2 || w.attempts >= 3 || w.requestAt != 0 && s.Frame-w.requestAt < 20 {
		return ""
	}
	w.requestAt = s.Frame
	w.reason = "empty_weapon"
	w.attempts++
	if w.attempts == 1 {
		if name, reason := economyWeapon(s); reason == "parasite_retreat_range" {
			return "use " + name
		}
		// Brief occlusion must not undo the distance constraint on depletion.
		// This retains a weapon constraint, not a fabricated visible target.
		if w.parasiteSeenAt > 0 && s.Frame-w.parasiteSeenAt <= 20 && s.InventoryKnown && s.InventoryAgeFrames <= 20 {
			for _, choice := range []struct{ name, ammo string }{{"Machinegun", "Bullets"}, {"HyperBlaster", "Cells"}} {
				if !strings.Contains(strings.ToLower(s.Weapon), weaponModel(choice.name)) && inventoryCount(s, choice.name) > 0 && inventoryCount(s, choice.ammo) > 0 {
					return "use " + choice.name
				}
			}
			return "use Blaster"
		}
		return stockedWeapon(s)
	}
	return "use Blaster"
}
