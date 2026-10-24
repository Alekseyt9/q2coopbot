package policy

import (
	"fmt"
	"math"
)

const MotorResidualVersion = "combat_observed_motor_residual_v1"
const MotorResidualInputWidth = 756

func validateMotorResidual(f PPOFile) error {
	if f.MotorResidual == nil {
		return nil
	}
	spec := f.MotorResidual
	if !equippedPrefixFeatures(f.Features) || f.AimModeHead != PrecisionHeadVersion || f.SpatialAim == nil || f.SampledMotionAim == nil || f.EntityAttention != nil || spec.Version != MotorResidualVersion || len(spec.Layers) != 3 {
		return fmt.Errorf("invalid motor residual contract")
	}
	width := MotorResidualInputWidth
	for i, output := range []int{64, 32, 2} {
		layer := spec.Layers[i]
		if len(layer.Weight) != output || len(layer.Bias) != output {
			return fmt.Errorf("invalid motor output width")
		}
		for r, row := range layer.Weight {
			if len(row) != width {
				return fmt.Errorf("invalid motor input width")
			}
			for _, value := range append(append([]float64(nil), row...), layer.Bias[r]) {
				if math.IsNaN(value) || math.IsInf(value, 0) || math.Abs(value) > 1e4 {
					return fmt.Errorf("invalid motor parameter")
				}
			}
		}
		width = output
	}
	return nil
}

// Only existing observed common/geometry/enemies, types, navigation and memory.
// Neither movement means nor future/native state are motor inputs.
func motorResidualInputs(features []float64) []float64 {
	x := append([]float64(nil), features[:169]...)
	x = append(x, features[466:786]...)
	x = append(x, features[854:1121]...)
	return x
}

func (p *PPO) motorOffsets(o Observation) ([2]float64, error) {
	var result [2]float64
	if p.file.MotorResidual == nil {
		return result, nil
	}
	features, err := FeaturesForVersion(o, EquippedWeaponFeatureVersion)
	if err != nil {
		return result, err
	}
	x := motorResidualInputs(features)
	for i, layer := range p.file.MotorResidual.Layers {
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
				return result, fmt.Errorf("nonfinite motor residual")
			}
		}
		x = y
	}
	copy(result[:], x)
	return result, nil
}

func (p *PPO) motorRaw(o Observation, raw []float64) ([]float64, error) {
	if p.file.MotorResidual == nil {
		return raw, nil
	}
	if len(raw) != PrecisionOutputWidth {
		return nil, fmt.Errorf("invalid motor raw width")
	}
	delta, err := p.motorOffsets(o)
	if err != nil {
		return nil, err
	}
	result := append([]float64(nil), raw...)
	for i := range delta {
		result[i] += delta[i]
		if math.IsNaN(result[i]) || math.IsInf(result[i], 0) {
			return nil, fmt.Errorf("nonfinite movement mean")
		}
	}
	return result, nil
}
