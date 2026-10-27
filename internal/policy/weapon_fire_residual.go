package policy

import (
	"fmt"
	"math"
)

const WeaponFireResidualVersion = "combat_observed_shotgun_fire_residual_v1"
const WeaponCycleFireResidualVersion = "combat_observed_handgrenade_hold11_fire_residual_v1"

func validateWeaponFireResidual(f PPOFile) error {
	if f.WeaponFireResidual == nil {
		return nil
	}
	if f.ConditionalFire == nil || (f.WeaponFireResidual.Version != WeaponFireResidualVersion && f.WeaponFireResidual.Version != WeaponCycleFireResidualVersion) {
		return fmt.Errorf("invalid weapon fire residual contract")
	}
	shape := *f.WeaponFireResidual
	shape.Version = ConditionalFireVersion
	if f.WeaponFireResidual.Version == WeaponCycleFireResidualVersion {
		shape.Version = ConditionalFireContextVersion
	}
	return validateConditionalFireSpec(&shape)
}

// This selects the applicable learned head; it does not choose attack/target.
func weaponFireResidualActive(features []float64) bool {
	return len(features) >= EquippedWeaponFeatureWidth && features[0] > 0 && features[1123] == 1
}

func weaponFireResidualActiveForVersion(features []float64, version string) bool {
	if version == WeaponCycleFireResidualVersion {
		return len(features) >= EquippedWeaponFeatureWidth && features[0] > 0 && features[1132] == 1 && math.Abs(features[5]-11.0/30) < 1e-6
	}
	return version == WeaponFireResidualVersion && weaponFireResidualActive(features)
}
