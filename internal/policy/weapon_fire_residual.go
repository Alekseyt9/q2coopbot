package policy

import "fmt"

const WeaponFireResidualVersion = "combat_observed_shotgun_fire_residual_v1"

func validateWeaponFireResidual(f PPOFile) error {
	if f.WeaponFireResidual == nil {
		return nil
	}
	if f.ConditionalFire == nil || f.WeaponFireResidual.Version != WeaponFireResidualVersion {
		return fmt.Errorf("invalid weapon fire residual contract")
	}
	shape := *f.WeaponFireResidual
	shape.Version = ConditionalFireVersion
	return validateConditionalFireSpec(&shape)
}

// This selects the applicable learned head; it does not choose attack/target.
func weaponFireResidualActive(features []float64) bool {
	return len(features) >= EquippedWeaponFeatureWidth && features[0] > 0 && features[1123] == 1
}
