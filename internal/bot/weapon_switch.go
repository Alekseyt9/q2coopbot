package bot

import "q2coopbot/internal/quake"

// Server frames bound retries independently of wall-clock acceleration.
// weapnext delegates inventory/ammo validation to the authoritative server.
type weaponSwitch struct {
	mapName                                 string
	lastFrame, emptyAt, requestAt, attempts int
}

func (w *weaponSwitch) command(s quake.Snapshot) string {
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
	if s.Weapon == "Blaster" || s.Ammo > 0 {
		w.emptyAt, w.requestAt, w.attempts = 0, 0, 0
		return ""
	}
	if w.emptyAt == 0 {
		w.emptyAt = s.Frame
	}
	if s.Frame-w.emptyAt < 2 || w.attempts >= 3 || w.requestAt != 0 && s.Frame-w.requestAt < 20 {
		return ""
	}
	w.requestAt = s.Frame
	w.attempts++
	if w.attempts == 1 {
		return stockedWeapon(s)
	}
	return "use Blaster"
}
