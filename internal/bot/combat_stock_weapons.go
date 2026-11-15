package bot

import "q2coopbot/internal/policy"

// Natural sessions retain the model's execution metadata and existing partner
// guards. Fixture and diagnostic overrides must never enter a human session.
func naturalStockExecution(c Config, contract string) bool {
	if contract != policy.StockWeaponsFireCombatExecution || c.CombatMode != "learned" ||
		!c.CombatCheckedWallSlide || c.CombatActionQualityGuard || c.CombatDiagnosticNominalAim {
		return false
	}
	if c.CombatLiveCompanion {
		return validateLiveLearnedCompanion(c) == nil
	}
	return c.TestCampaignCombatEvaluation && validateCampaignCombatEvaluation(c) == nil
}

func stockWeaponNeedsSolo(weapon string) bool {
	return weapon != "Blaster" && weapon != "models/weapons/v_blast/tris.md2" && !directHitscanWeapon(weapon)
}

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
