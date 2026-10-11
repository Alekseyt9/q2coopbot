package policy

import (
	"fmt"
	"math"
)

const WeaponStateResidualVersion = "combat_observed_weapon_state_residual_v1"
const WeaponStateResidualInputWidth = 154

func validateWeaponStateResidual(f PPOFile) error {
	if f.WeaponStateResidual == nil {
		return nil
	}
	spec := f.WeaponStateResidual
	if f.Features != EquippedWeaponFeatureVersion || f.AimModeHead != PrecisionHeadVersion || f.WeaponHead != WeaponHeadVersion || f.TargetHead != TargetHeadVersion || f.SpatialAim == nil || f.EntityAttention != nil || spec.Version != WeaponStateResidualVersion || len(spec.Layers) != 3 {
		return fmt.Errorf("invalid observed weapon residual contract")
	}
	width := WeaponStateResidualInputWidth
	for i, output := range []int{64, 32, 12} {
		layer := spec.Layers[i]
		if len(layer.Bias) != output || len(layer.Weight) != output {
			return fmt.Errorf("invalid weapon residual output width")
		}
		for r, row := range layer.Weight {
			if len(row) != width {
				return fmt.Errorf("invalid weapon residual input width")
			}
			if math.IsNaN(layer.Bias[r]) || math.IsInf(layer.Bias[r], 0) || math.Abs(layer.Bias[r]) > 1e4 {
				return fmt.Errorf("invalid weapon residual bias")
			}
			for _, value := range row {
				if math.IsNaN(value) || math.IsInf(value, 0) || math.Abs(value) > 1e4 {
					return fmt.Errorf("invalid weapon residual weight")
				}
			}
		}
		width = output
	}
	return nil
}

// Only observed state and the existing masked soft target context enter this
// branch. The model's weapon availability and all other heads remain unchanged.
func weaponStateInputs(features, probabilities []float64) []float64 {
	x := append([]float64(nil), features[:20]...)
	x = append(x, features[814:845]...)
	x = append(x, features[1121:1147]...)
	for _, block := range []struct{ start, width int }{{73, 12}, {466, 40}} {
		selected := make([]float64, block.width)
		for slot := 0; slot < 8; slot++ {
			for j := range selected {
				selected[j] += probabilities[slot+1] * features[block.start+slot*block.width+j]
			}
		}
		x = append(x, selected...)
	}
	mean, maximum := targetSummary(features[73:169], 12)
	x = append(x, mean...)
	x = append(x, maximum...)
	return append(x, probabilities[0])
}

func (p *PPO) weaponStateRaw(o Observation, raw []float64) ([]float64, error) {
	if p.file.WeaponStateResidual == nil {
		return raw, nil
	}
	if len(raw) != PrecisionOutputWidth {
		return nil, fmt.Errorf("invalid weapon residual raw width")
	}
	features, err := FeaturesForVersion(o, EquippedWeaponFeatureVersion)
	if err != nil {
		return nil, err
	}
	logProbabilities, err := targetLogProbabilities(o, raw[TargetLogitOffset:TargetAimOffset])
	if err != nil {
		return nil, err
	}
	probabilities := make([]float64, len(logProbabilities))
	for i, value := range logProbabilities {
		probabilities[i] = math.Exp(value)
	}
	x := weaponStateInputs(features, probabilities)
	for index, layer := range p.file.WeaponStateResidual.Layers {
		y := make([]float64, len(layer.Bias))
		for r, row := range layer.Weight {
			y[r] = layer.Bias[r]
			for c, weight := range row {
				y[r] += weight * x[c]
			}
			if index < 2 {
				y[r] = math.Max(0, y[r])
			}
			if math.IsNaN(y[r]) || math.IsInf(y[r], 0) {
				return nil, fmt.Errorf("nonfinite weapon residual output")
			}
		}
		x = y
	}
	result := append([]float64(nil), raw...)
	for i, value := range x {
		result[8+i] += value
		if math.IsNaN(result[8+i]) || math.IsInf(result[8+i], 0) {
			return nil, fmt.Errorf("nonfinite weapon logit")
		}
	}
	return result, nil
}
