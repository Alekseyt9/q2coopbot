package policy

import "fmt"

// Weapon-only context preserves all v6 world/motor inputs and appends the
// existing observed-equipped/own-request encoding used by v10 and later PPO.
const HJEPAWeaponFeatureVersion = "hjepa_weapon_features_v1"
const HJEPAWeaponFeatureWidth = WeaponFeatureWidth + 26

// HJEPAContextFeatureVersion exposes the same causal context to world and motor
// networks. Its declaration is independent of the optional weapon head format.
const HJEPAContextFeatureVersion = "hjepa_context_features_v1"

func HJEPAFeatureWidth(version string) (int, error) {
	switch version {
	case WeaponFeatureVersion:
		return WeaponFeatureWidth, nil
	case HJEPAContextFeatureVersion:
		return HJEPAWeaponFeatureWidth, nil
	default:
		return 0, fmt.Errorf("unsupported H-JEPA features")
	}
}

func HJEPAFeatures(o Observation, version string) ([]float64, error) {
	switch version {
	case WeaponFeatureVersion:
		return FeaturesForVersion(o, version)
	case HJEPAContextFeatureVersion:
		return HJEPAWeaponFeatures(o)
	default:
		return nil, fmt.Errorf("unsupported H-JEPA features")
	}
}

func appendHJEPAWeaponContext(o Observation, base []float64) ([]float64, error) {
	tail, err := equippedWeaponFeatures(o)
	if err != nil {
		return nil, err
	}
	out := append([]float64(nil), base...)
	return append(out, tail...), nil
}

func HJEPAWeaponFeatures(o Observation) ([]float64, error) {
	base, err := FeaturesForVersion(o, WeaponFeatureVersion)
	if err != nil {
		return nil, err
	}
	return appendHJEPAWeaponContext(o, base)
}

func (h *HJEPA) weaponInputWidth() int {
	if h.file.WeaponFeatures == HJEPAWeaponFeatureVersion {
		return HJEPAWeaponFeatureWidth
	}
	return WeaponFeatureWidth
}
