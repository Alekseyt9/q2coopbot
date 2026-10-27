package policy

import (
	"fmt"
	"math"
)

// Optional fire residual conditioned on the complete sampled action, before
// execution guards or quantization. No engine outcomes are model inputs.
const ConditionalFireVersion = "combat_action_conditioned_fire_v1"
const ConditionalFireInputWidth = 181
const ConditionalFireContextVersion = "combat_action_conditioned_fire_context_v2"
const ConditionalFireContextInputWidth = ConditionalFireInputWidth + 49 + 37

func validateConditionalFire(f PPOFile) error {
	if err := validateWeaponFireResidual(f); err != nil {
		return err
	}
	if f.ConditionalFire == nil {
		return nil
	}
	if !equippedPrefixFeatures(f.Features) || f.AimModeHead != PrecisionHeadVersion || f.SpatialAim == nil || f.SampledMotionAim == nil || f.EntityAttention != nil {
		return fmt.Errorf("invalid conditional fire contract")
	}
	return validateConditionalFireSpec(f.ConditionalFire)
}

func (p *PPO) conditionalFireLogit(o Observation, base float64, s Sample) (float64, error) {
	if p.file.ConditionalFire == nil {
		return base, nil
	}
	features, err := FeaturesForVersion(o, EquippedWeaponFeatureVersion)
	if err != nil {
		return 0, err
	}
	x, err := conditionalFireInputsForVersion(features, s, p.file.ConditionalFire.Version)
	if err != nil {
		return 0, err
	}
	for i, layer := range p.file.ConditionalFire.Layers {
		y := make([]float64, len(layer.Bias))
		for r, row := range layer.Weight {
			y[r] = layer.Bias[r]
			for c, weight := range row {
				y[r] += weight * x[c]
			}
			if i < 2 {
				y[r] = math.Max(0, y[r])
			}
			if math.IsNaN(y[r]) || math.IsInf(y[r], 0) {
				return 0, fmt.Errorf("nonfinite conditional fire residual")
			}
		}
		x = y
	}
	result := base + x[0]
	if p.file.WeaponFireResidual != nil && weaponFireResidualActiveForVersion(features, p.file.WeaponFireResidual.Version) {
		version := ConditionalFireVersion
		if p.file.WeaponFireResidual.Version == WeaponCycleFireResidualVersion {
			version = ConditionalFireContextVersion
		}
		inputs, err := conditionalFireInputsForVersion(features, s, version)
		if err != nil {
			return 0, err
		}
		for i, layer := range p.file.WeaponFireResidual.Layers {
			y := make([]float64, len(layer.Bias))
			for r, row := range layer.Weight {
				y[r] = layer.Bias[r]
				for c, weight := range row {
					y[r] += weight * inputs[c]
				}
				if i < 2 {
					y[r] = math.Max(0, y[r])
				}
				if math.IsNaN(y[r]) || math.IsInf(y[r], 0) {
					return 0, fmt.Errorf("nonfinite weapon fire residual")
				}
			}
			inputs = y
		}
		result += inputs[0]
	}
	if math.IsNaN(result) || math.IsInf(result, 0) {
		return 0, fmt.Errorf("nonfinite conditional fire logit")
	}
	return result, nil
}

func validateConditionalFireSpec(spec *SpatialAimFile) error {
	if spec == nil {
		return nil
	}
	if (spec.Version != ConditionalFireVersion && spec.Version != ConditionalFireContextVersion) || len(spec.Layers) != 3 {
		return fmt.Errorf("invalid conditional fire version/layers")
	}
	width := ConditionalFireInputWidth
	if spec.Version == ConditionalFireContextVersion {
		width = ConditionalFireContextInputWidth
	}
	for i, output := range []int{32, 16, 1} {
		layer := spec.Layers[i]
		if len(layer.Weight) != output || len(layer.Bias) != output {
			return fmt.Errorf("invalid conditional fire output width")
		}
		for r, row := range layer.Weight {
			if len(row) != width {
				return fmt.Errorf("invalid conditional fire input width")
			}
			for _, value := range append(append([]float64(nil), row...), layer.Bias[r]) {
				if math.IsNaN(value) || math.IsInf(value, 0) || math.Abs(value) > 1e4 {
					return fmt.Errorf("invalid conditional fire parameter")
				}
			}
		}
		width = output
	}
	return nil
}

func conditionalFireInputs(features []float64, s Sample) ([]float64, error) {
	if len(features) != 1147 || s.Target < 0 || s.Target > 8 || s.AimMode < 0 || s.AimMode > 1 || s.Vertical < 0 || s.Vertical > 2 || s.Weapon < 0 || s.Weapon > 11 {
		return nil, fmt.Errorf("invalid conditional fire context")
	}
	for _, value := range features {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return nil, fmt.Errorf("nonfinite fire feature")
		}
	}
	for slot := 0; slot < 8; slot++ {
		mask := features[426+5*slot]
		if mask != 0 && mask != 1 {
			return nil, fmt.Errorf("invalid target availability mask")
		}
	}
	for slot := 0; slot < 12; slot++ {
		mask := features[WeaponMaskOffset+slot]
		if mask != 0 && mask != 1 {
			return nil, fmt.Errorf("invalid weapon availability mask")
		}
	}
	if features[WeaponMaskOffset] != 1 || features[WeaponMaskOffset+s.Weapon] != 1 || s.Target > 0 && features[426+5*(s.Target-1)] != 1 {
		return nil, fmt.Errorf("unavailable conditional fire target/weapon")
	}
	x := append([]float64(nil), features[:73]...)
	x = append(x, features[1121:1147]...)
	if s.Target > 0 {
		slot := s.Target - 1
		x = append(x, features[73+slot*12:73+(slot+1)*12]...)
		x = append(x, features[466+slot*40:466+(slot+1)*40]...)
	} else {
		x = append(x, make([]float64, 52)...)
	}
	onehot := func(category, width int) {
		v := make([]float64, width)
		v[category] = 1
		x = append(x, v...)
	}
	onehot(s.Target, 9)
	onehot(s.AimMode, 2)
	for i, value := range s.Latent {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return nil, fmt.Errorf("nonfinite sampled fire action")
		}
		physical := math.Tanh(value)
		if i >= 2 && s.AimMode == 1 {
			physical *= PrecisionFineDegrees / 180.
		}
		x = append(x, physical)
	}
	onehot(s.Vertical, 3)
	onehot(s.Weapon, 12)
	return x, nil
}

func conditionalFireInputsForVersion(features []float64, s Sample, version string) ([]float64, error) {
	if version != ConditionalFireVersion && version != ConditionalFireContextVersion {
		return nil, fmt.Errorf("unknown conditional fire input version")
	}
	x, err := conditionalFireInputs(features, s)
	if err != nil {
		return nil, err
	}
	if version == ConditionalFireContextVersion {
		x = append(x, features[169:218]...)
		x = append(x, features[259:296]...)
	}
	return x, nil
}
