package policy

import (
	"fmt"
	"math"
)

const WeaponExplorationVersion = "combat_observed_grenades_weapon_temperature_v1"
const BlasterWeaponExplorationVersion = "combat_observed_blaster_grenades_weapon_temperature_v2"

type WeaponExplorationFile struct {
	Version            string   `json:"version"`
	Temperature        float64  `json:"temperature"`
	BlasterTemperature *float64 `json:"blaster_temperature,omitempty"`
}

func validateWeaponExploration(f PPOFile) error {
	if f.WeaponExploration == nil {
		return nil
	}
	spec := f.WeaponExploration
	validVersion := spec.Version == WeaponExplorationVersion && spec.BlasterTemperature == nil ||
		spec.Version == BlasterWeaponExplorationVersion && spec.BlasterTemperature != nil &&
			!math.IsNaN(*spec.BlasterTemperature) && !math.IsInf(*spec.BlasterTemperature, 0) && *spec.BlasterTemperature >= 1 && *spec.BlasterTemperature <= 4
	if !validVersion || !equippedPrefixFeatures(f.Features) || f.WeaponHead != WeaponHeadVersion || f.AimModeHead != PrecisionHeadVersion || f.SpatialAim == nil || f.EntityAttention != nil || math.IsNaN(spec.Temperature) || math.IsInf(spec.Temperature, 0) || spec.Temperature < 1 || spec.Temperature > 4 {
		return fmt.Errorf("invalid observed weapon exploration contract")
	}
	return nil
}

// Exploration only changes the final masked weapon distribution. The observed
// equipped weapon and fresh inventory gate it; every choice remains available
// under its original mask, and temperature preserves deterministic argmax.
func (p *PPO) weaponExplorationRaw(o Observation, raw []float64) ([]float64, error) {
	spec := p.file.WeaponExploration
	if spec == nil {
		return raw, nil
	}
	equipped := equippedWeaponIndex(o.Weapon)
	temperature := spec.Temperature
	if equipped == 1 && spec.Version == BlasterWeaponExplorationVersion && spec.BlasterTemperature != nil {
		temperature = *spec.BlasterTemperature
	} else if equipped != 11 {
		return raw, nil
	}
	if temperature == 1 {
		return raw, nil
	}
	if len(raw) != PrecisionOutputWidth {
		return nil, fmt.Errorf("invalid weapon exploration output width")
	}
	mask, err := WeaponAvailability(o)
	if err != nil {
		return nil, err
	}
	enabled := mask[1] || mask[2] || mask[3] || mask[4]
	if equipped == 1 {
		enabled = false
		for _, available := range mask[2:] {
			enabled = enabled || available
		}
	}
	if !enabled {
		return raw, nil
	}
	result := append([]float64(nil), raw...)
	for i := 8; i < 20; i++ {
		result[i] /= temperature
	}
	return result, nil
}
