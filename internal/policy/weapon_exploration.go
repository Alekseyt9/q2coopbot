package policy

import (
	"fmt"
	"math"
)

const WeaponExplorationVersion = "combat_observed_grenades_weapon_temperature_v1"

type WeaponExplorationFile struct {
	Version     string  `json:"version"`
	Temperature float64 `json:"temperature"`
}

func validateWeaponExploration(f PPOFile) error {
	if f.WeaponExploration == nil {
		return nil
	}
	spec := f.WeaponExploration
	if spec.Version != WeaponExplorationVersion || f.Features != EquippedWeaponFeatureVersion || f.WeaponHead != WeaponHeadVersion || f.AimModeHead != PrecisionHeadVersion || f.SpatialAim == nil || f.EntityAttention != nil || math.IsNaN(spec.Temperature) || math.IsInf(spec.Temperature, 0) || spec.Temperature < 1 || spec.Temperature > 4 {
		return fmt.Errorf("invalid observed weapon exploration contract")
	}
	return nil
}

// Exploration only changes the final masked weapon distribution. The observed
// equipped weapon and fresh inventory gate it; every choice remains available
// under its original mask, and temperature preserves deterministic argmax.
func (p *PPO) weaponExplorationRaw(o Observation, raw []float64) ([]float64, error) {
	spec := p.file.WeaponExploration
	if spec == nil || spec.Temperature == 1 || equippedWeaponIndex(o.Weapon) != 11 {
		return raw, nil
	}
	if len(raw) != PrecisionOutputWidth {
		return nil, fmt.Errorf("invalid weapon exploration output width")
	}
	mask, err := WeaponAvailability(o)
	if err != nil {
		return nil, err
	}
	if !mask[1] && !mask[2] && !mask[3] && !mask[4] {
		return raw, nil
	}
	result := append([]float64(nil), raw...)
	for i := 8; i < 20; i++ {
		result[i] /= spec.Temperature
	}
	return result, nil
}
