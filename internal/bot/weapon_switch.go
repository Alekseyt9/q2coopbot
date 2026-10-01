package bot

import "q2coopbot/internal/quake"

// Server frames bound retries independently of wall-clock acceleration.
// weapnext delegates inventory/ammo validation to the authoritative server.
type weaponSwitch struct {
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
		return stockedWeapon(s)
	}
	return "use Blaster"
}
