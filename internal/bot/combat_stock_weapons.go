package bot

import "q2coopbot/internal/policy"

// Exact stock names/models only; an unknown mod weapon remains unsupported.
func directStockWeapon(weapon string) bool {
	models := [...]string{"v_blast", "v_shotg", "v_shotg2", "v_machn", "v_chain", "v_launch", "v_rocket", "v_hyperb", "v_rail", "v_bfg", "v_handgr"}
	for i, name := range policy.WeaponChoices()[1:] {
		if weapon == name || weapon == "models/weapons/"+models[i]+"/tris.md2" {
			return true
		}
	}
	return false
}
