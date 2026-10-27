package bot

import (
	"q2coopbot/internal/policy"
	"q2coopbot/internal/quake"
)

// Acquisition-only intervention. The caller admits exactly one captured local
// first-life frame. Keep the sampled policy request; change only native attack.
func diagnosticGrenadeRelease(s quake.Snapshot, guarded quake.UserCmd) (quake.UserCmd, []policy.Intervention) {
	if s.Health <= 0 || s.GunFrame != 11 || (s.Weapon != "Grenades" && s.Weapon != "models/weapons/v_handgr/tris.md2") || guarded.Buttons&1 == 0 {
		return guarded, nil
	}
	guarded.Buttons &^= 1
	return guarded, []policy.Intervention{{Component: "attack", Reason: "diagnostic_grenade_release"}}
}
