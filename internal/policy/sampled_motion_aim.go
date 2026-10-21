package policy

import (
	"fmt"
	"math"
)

const SampledMotionAimVersion = "combat_sampled_motion_aim_v1"
const SampledMotionAimInputWidth = SpatialAimInputWidth + 6

func validateSampledMotionAim(f PPOFile) error {
	spec := f.SampledMotionAim
	if spec == nil {
		return nil
	}
	if spec.Version != SampledMotionAimVersion || f.Features != EquippedWeaponFeatureVersion || f.SpatialAim == nil || f.AimModeHead != PrecisionHeadVersion || len(spec.Layers) != 3 {
		return fmt.Errorf("invalid sampled motion aim contract")
	}
	width := SampledMotionAimInputWidth
	for i, output := range []int{64, 32, 2} {
		layer := spec.Layers[i]
		if len(layer.Weight) != output || len(layer.Bias) != output {
			return fmt.Errorf("invalid sampled motion aim output width")
		}
		for r, row := range layer.Weight {
			if len(row) != width {
				return fmt.Errorf("invalid sampled motion aim input width")
			}
			for _, v := range append(append([]float64(nil), row...), layer.Bias[r]) {
				if math.IsNaN(v) || math.IsInf(v, 0) || math.Abs(v) > 1e4 {
					return fmt.Errorf("invalid sampled motion aim parameter")
				}
			}
		}
		width = output
	}
	return nil
}

// Sampled movement is the declared command, not future displacement after guards.
// This branch changes only yaw/pitch means conditional on target, mode and move.
func (p *PPO) sampledMotionAimMeans(o Observation, raw []float64, means [4]float64, movement [2]float64, target, mode int) ([4]float64, error) {
	if p.file.SampledMotionAim == nil || target == 0 {
		return means, nil
	}
	if target < 1 || target > 8 || mode < 0 || mode > 1 || len(raw) != PrecisionOutputWidth {
		return means, fmt.Errorf("invalid sampled motion aim selection")
	}
	features, err := FeaturesForVersion(o, p.file.Features)
	if err != nil {
		return means, err
	}
	x := spatialAimInputs(features, raw, target-1)
	fwd, side := math.Tanh(movement[0]), math.Tanh(movement[1])
	x = append(x, fwd, side, fwd-math.Tanh(raw[0]), side-math.Tanh(raw[1]), float64(1-mode), float64(mode))
	for i, layer := range p.file.SampledMotionAim.Layers {
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
				return means, fmt.Errorf("nonfinite sampled motion aim output")
			}
		}
		x = y
	}
	means[2] += x[0]
	means[3] += x[1]
	return means, nil
}
