package policy

import (
	"fmt"
	"math"
)

const MemoryRecoveryVersion = "combat_memory_coarse_recovery_v1"
const MemoryRecoveryInputWidth = 46

func validateMemoryRecovery(f PPOFile) error {
	if f.MemoryRecovery == nil {
		return nil
	}
	spec := f.MemoryRecovery
	if f.Features != EquippedWeaponFeatureVersion || f.AimModeHead != PrecisionHeadVersion || f.SpatialAim == nil || f.EntityAttention != nil || spec.Version != MemoryRecoveryVersion || len(spec.Layers) != 3 {
		return fmt.Errorf("invalid memory recovery contract")
	}
	width := MemoryRecoveryInputWidth
	for i, output := range []int{32, 16, 3} {
		layer := spec.Layers[i]
		if len(layer.Weight) != output || len(layer.Bias) != output {
			return fmt.Errorf("invalid memory recovery output width")
		}
		for r, row := range layer.Weight {
			if len(row) != width {
				return fmt.Errorf("invalid memory recovery input width")
			}
			for _, v := range append(append([]float64(nil), row...), layer.Bias[r]) {
				if math.IsNaN(v) || math.IsInf(v, 0) || math.Abs(v) > 1e4 {
					return fmt.Errorf("invalid memory recovery parameter")
				}
			}
		}
		width = output
	}
	return nil
}

func memoryRecoveryInputs(features, raw []float64) ([]float64, bool, error) {
	if len(features) != EquippedWeaponFeatureWidth || len(raw) != PrecisionOutputWidth {
		return nil, false, fmt.Errorf("invalid memory recovery dimensions")
	}
	active := features[881] == 1 && features[882] <= 10.0/200
	for slot := 0; slot < 8; slot++ {
		if features[73+slot*12] != 0 {
			active = false
		}
	}
	x := append([]float64(nil), features[:13]...)
	x = append(x, features[881:911]...)
	for _, v := range raw[2:5] {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return nil, false, fmt.Errorf("nonfinite memory recovery raw")
		}
		x = append(x, math.Max(-20, math.Min(20, v)))
	}
	for _, v := range x {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return nil, false, fmt.Errorf("nonfinite memory recovery input")
		}
	}
	return x, active, nil
}

// A state-only residual at the final actor-raw stage; no sampling draws or
// temporal-memory changes. Visibility and memory are client observations.
func (p *PPO) memoryRecoveryRaw(o Observation, raw []float64) ([]float64, error) {
	if p.file.MemoryRecovery == nil {
		return raw, nil
	}
	features, err := FeaturesForVersion(o, EquippedWeaponFeatureVersion)
	if err != nil {
		return nil, err
	}
	x, active, err := memoryRecoveryInputs(features, raw)
	if err != nil || !active {
		return raw, err
	}
	for i, layer := range p.file.MemoryRecovery.Layers {
		y := make([]float64, len(layer.Bias))
		for r, row := range layer.Weight {
			y[r] = layer.Bias[r]
			for c, w := range row {
				y[r] += w * x[c]
			}
			if i < 2 {
				y[r] = math.Max(0, y[r])
			}
			if math.IsNaN(y[r]) || math.IsInf(y[r], 0) {
				return nil, fmt.Errorf("nonfinite memory recovery output")
			}
		}
		x = y
	}
	result := append([]float64(nil), raw...)
	result[2] += x[0]
	result[3] += x[1]
	if features[1125] == 1 { // Equipped machinegun, not inventory or requested use.
		result[4] += x[2]
	}
	for _, v := range result[2:5] {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return nil, fmt.Errorf("nonfinite recovered actor raw")
		}
	}
	return result, nil
}
