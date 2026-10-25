package policy

import (
	"fmt"
	"math"
)

const MemoryRouteMovementVersion = "combat_memory_route_movement_v1"
const MemoryRouteMovementInputWidth = 13 + 240 + MemoryRouteTailWidth + 5

func validateMemoryRouteMovement(f PPOFile) error {
	s := f.MemoryRouteMovement
	if s == nil {
		if f.Features == MemoryRouteFeatureVersion {
			return fmt.Errorf("route features require versioned movement branch")
		}
		return nil
	}
	if f.Features != MemoryRouteFeatureVersion || s.Version != MemoryRouteMovementVersion || f.SpatialAim == nil || f.MemoryRecovery != nil || f.MovementInertia != nil || f.EntityAttention != nil || len(s.Layers) != 3 {
		return fmt.Errorf("invalid memory route movement contract")
	}
	width := MemoryRouteMovementInputWidth
	for i, out := range []int{32, 16, 2} {
		l := s.Layers[i]
		if len(l.Weight) != out || len(l.Bias) != out {
			return fmt.Errorf("invalid route layer output")
		}
		for j, row := range l.Weight {
			if len(row) != width {
				return fmt.Errorf("invalid route layer input")
			}
			for _, v := range append(append([]float64{}, row...), l.Bias[j]) {
				if math.IsNaN(v) || math.IsInf(v, 0) || math.Abs(v) > 1e4 {
					return fmt.Errorf("invalid route parameter")
				}
			}
		}
		width = out
	}
	return nil
}

func memoryRouteMovementInputs(x, raw []float64) ([]float64, bool, error) {
	if len(x) != MemoryRouteFeatureWidth || len(raw) != PrecisionOutputWidth {
		return nil, false, fmt.Errorf("invalid route input width")
	}
	active := x[0] > 0 && x[ProjectileTypeFeatureWidth] == 1
	found := false
	for i := 0; i < 8; i++ {
		if x[73+i*12] != 0 {
			active = false
		}
		at := ProjectileTypeFeatureWidth + 1 + i*8
		if x[881+i*30] == 1 && x[at] == 1 && x[at+1] == 1 {
			found = true
		}
	}
	input := append([]float64{}, x[:13]...)
	input = append(input, x[881:1121]...)
	input = append(input, x[ProjectileTypeFeatureWidth:]...)
	for _, v := range raw[:5] {
		input = append(input, math.Max(-20, math.Min(20, v)))
	}
	for _, v := range input {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return nil, false, fmt.Errorf("nonfinite route input")
		}
	}
	return input, active && found, nil
}

func (p *PPO) memoryRouteMovementRaw(o Observation, raw []float64) ([]float64, error) {
	if p.file.MemoryRouteMovement == nil {
		return raw, nil
	}
	features, err := FeaturesForVersion(o, MemoryRouteFeatureVersion)
	if err != nil {
		return nil, err
	}
	x, active, err := memoryRouteMovementInputs(features, raw)
	if err != nil || !active {
		return raw, err
	}
	for i, l := range p.file.MemoryRouteMovement.Layers {
		y := affine(l, x)
		for j, v := range y {
			if i < 2 {
				y[j] = math.Max(0, v)
			}
			if math.IsNaN(y[j]) || math.IsInf(y[j], 0) {
				return nil, fmt.Errorf("nonfinite route movement")
			}
		}
		x = y
	}
	result := append([]float64{}, raw...)
	result[0] += x[0]
	result[1] += x[1]
	return result, nil
}
