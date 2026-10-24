package policy

import (
	"fmt"
	"math"
)

const SampledMotionAimVersion = "combat_sampled_motion_aim_v1"
const SampledVerticalAimVersion = "combat_sampled_motion_vertical_aim_v2"
const SampledVerticalControlVersion = "combat_sampled_motion_vertical_ablated_v2"
const SampledMotionAimInputWidth = SpatialAimInputWidth + 6

func validateSampledMotionAim(f PPOFile) error {
	spec := f.SampledMotionAim
	if spec == nil {
		return nil
	}
	if (spec.Version != SampledMotionAimVersion && spec.Version != SampledVerticalAimVersion && spec.Version != SampledVerticalControlVersion) || !equippedPrefixFeatures(f.Features) || f.SpatialAim == nil || f.AimModeHead != PrecisionHeadVersion || len(spec.Layers) != 3 {
		return fmt.Errorf("invalid sampled motion aim contract")
	}
	width := SampledMotionAimInputWidth
	if spec.Version != SampledMotionAimVersion {
		width += 3
	}
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
func (p *PPO) sampledMotionAimMeans(o Observation, raw []float64, means [4]float64, movement [2]float64, target, mode, vertical int) ([4]float64, error) {
	if p.file.SampledMotionAim == nil || target == 0 {
		return means, nil
	}
	if target < 1 || target > 8 || mode < 0 || mode > 1 || vertical < 0 || vertical > 2 || len(raw) != PrecisionOutputWidth {
		return means, fmt.Errorf("invalid sampled motion aim selection")
	}
	features, err := FeaturesForVersion(o, p.file.Features)
	if err != nil {
		return means, err
	}
	if p.file.MotorResidual != nil {
		delta, e := p.motorOffsets(o)
		if e != nil {
			return means, e
		}
		raw = append([]float64(nil), raw...)
		raw[0] -= delta[0]
		raw[1] -= delta[1]
	}
	x := spatialAimInputs(features, raw, target-1)
	fwd, side := math.Tanh(movement[0]), math.Tanh(movement[1])
	x = append(x, fwd, side, fwd-math.Tanh(raw[0]), side-math.Tanh(raw[1]), float64(1-mode), float64(mode))
	if p.file.SampledMotionAim.Version != SampledMotionAimVersion {
		extra := [3]float64{}
		if p.file.SampledMotionAim.Version == SampledVerticalAimVersion {
			extra[vertical] = 1
		}
		x = append(x, extra[:]...)
	}
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
