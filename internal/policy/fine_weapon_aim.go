package policy

import (
	"fmt"
	"math"
)

const FineWeaponAimVersion = "combat_observed_shotgun_fine_aim_v1"
const FineWeaponAimInputWidth = SampledMotionAimInputWidth + 26 + 2

func validateFineWeaponAim(f PPOFile) error {
	s := f.FineWeaponAim
	if s == nil {
		return nil
	}
	if s.Version != FineWeaponAimVersion || !equippedPrefixFeatures(f.Features) || f.SampledMotionAim == nil || f.SampledMotionAim.Version != SampledMotionAimVersion || f.SpatialAim == nil || f.AimModeHead != PrecisionHeadVersion || len(s.Layers) != 3 {
		return fmt.Errorf("invalid fine weapon aim contract")
	}
	width := FineWeaponAimInputWidth
	for i, output := range []int{32, 16, 2} {
		l := s.Layers[i]
		if len(l.Weight) != output || len(l.Bias) != output {
			return fmt.Errorf("invalid fine weapon aim output width")
		}
		for row, weights := range l.Weight {
			if len(weights) != width {
				return fmt.Errorf("invalid fine weapon aim input width")
			}
			for _, v := range append(append([]float64(nil), weights...), l.Bias[row]) {
				if math.IsNaN(v) || math.IsInf(v, 0) || math.Abs(v) > 1e4 {
					return fmt.Errorf("invalid fine weapon aim parameter")
				}
			}
		}
		width = output
	}
	return nil
}

// Eligibility chooses which learned residual is applicable, never its aim or
// target. Equipped weapon, visible slot and target availability are observations.
func fineWeaponAimActive(features []float64, target, mode int) bool {
	if len(features) < EquippedWeaponFeatureWidth || target < 1 || target > 8 || mode != 1 {
		return false
	}
	slot := target - 1
	return features[0] > 0 && features[1123] == 1 && features[73+12*slot] == 1 && features[426+5*slot] == 1
}

func (p *PPO) fineWeaponAimMeans(features, motionInputs []float64, means [4]float64, target, mode int) ([4]float64, error) {
	if p.file.FineWeaponAim == nil || !fineWeaponAimActive(features, target, mode) {
		return means, nil
	}
	if len(motionInputs) != SampledMotionAimInputWidth {
		return means, fmt.Errorf("invalid fine weapon aim context")
	}
	x := append([]float64(nil), motionInputs...)
	x = append(x, features[1121:1147]...)
	x = append(x, math.Max(-8, math.Min(8, means[2])), math.Max(-8, math.Min(8, means[3])))
	for i, layer := range p.file.FineWeaponAim.Layers {
		y := make([]float64, len(layer.Bias))
		for r, weights := range layer.Weight {
			y[r] = layer.Bias[r]
			for c, w := range weights {
				y[r] += w * x[c]
			}
			if i < 2 {
				y[r] = math.Max(0, y[r])
			}
			if math.IsNaN(y[r]) || math.IsInf(y[r], 0) {
				return means, fmt.Errorf("nonfinite fine weapon aim output")
			}
		}
		x = y
	}
	means[2] += x[0]
	means[3] += x[1]
	return means, nil
}
