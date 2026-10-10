package policy

import (
	"fmt"
	"math"
)

const SharedTargetVersion = "combat_shared_target_residual_v1"
const SharedTargetInputWidth = 219

func validateSharedTarget(f PPOFile) error {
	if f.SharedTarget == nil {
		return nil
	}
	spec := f.SharedTarget
	if f.Features != ThreatFeatureVersion || f.SpatialAim == nil || f.AimModeHead != PrecisionHeadVersion || spec.Version != SharedTargetVersion || len(spec.Layers) != 3 {
		return fmt.Errorf("invalid shared target contract")
	}
	width := SharedTargetInputWidth
	for i, output := range []int{64, 32, 1} {
		layer := spec.Layers[i]
		if len(layer.Bias) != output || len(layer.Weight) != output {
			return fmt.Errorf("invalid shared target output width")
		}
		for r, row := range layer.Weight {
			if len(row) != width {
				return fmt.Errorf("invalid shared target input width")
			}
			for _, v := range append(append([]float64(nil), row...), layer.Bias[r]) {
				if math.IsNaN(v) || math.IsInf(v, 0) || math.Abs(v) > 1e4 {
					return fmt.Errorf("invalid shared target parameter")
				}
			}
		}
		width = output
	}
	return nil
}

// Pool observed slots only. Empty slots do not affect means or maxima.
func targetSummary(values []float64, width int) ([]float64, []float64) {
	mean, maximum := make([]float64, width), make([]float64, width)
	count := 0
	for slot := 0; slot < 8; slot++ {
		x := values[slot*width : (slot+1)*width]
		if x[0] <= 0 {
			continue
		}
		for i, v := range x {
			mean[i] += v
			if count == 0 || v > maximum[i] {
				maximum[i] = v
			}
		}
		count++
	}
	if count > 0 {
		for i := range mean {
			mean[i] /= float64(count)
		}
	}
	return mean, maximum
}

func sharedTargetInputs(features, raw []float64, slot int) []float64 {
	x := spatialAimInputs(features, raw, slot)
	x = append(x, features[846+slot])
	x = append(x, features[854:881]...)
	mean, maximum := targetSummary(features[881:1121], 30)
	x = append(x, mean...)
	x = append(x, maximum...)
	group, _ := targetSummary(features[73:169], 12)
	return append(x, group...)
}

func (p *PPO) sharedTargetRaw(o Observation, raw []float64) ([]float64, error) {
	if p.file.SharedTarget == nil {
		return raw, nil
	}
	features, err := FeaturesForVersion(o, ThreatFeatureVersion)
	if err != nil {
		return nil, err
	}
	result := append([]float64(nil), raw...)
	for slot := 0; slot < 8; slot++ {
		x := sharedTargetInputs(features, raw, slot)
		for index, layer := range p.file.SharedTarget.Layers {
			y := make([]float64, len(layer.Bias))
			for r, row := range layer.Weight {
				y[r] = layer.Bias[r]
				for c, w := range row {
					y[r] += w * x[c]
				}
				if index < 2 {
					y[r] = math.Max(0, y[r])
				}
				if math.IsNaN(y[r]) || math.IsInf(y[r], 0) {
					return nil, fmt.Errorf("nonfinite shared target output")
				}
			}
			x = y
		}
		result[TargetLogitOffset+slot+1] += x[0]
	}
	return result, nil
}
