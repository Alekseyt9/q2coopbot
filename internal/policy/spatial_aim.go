package policy

import (
	"fmt"
	"math"
)

const SpatialAimVersion = "combat_shared_spatial_fine_aim_v1"
const SpatialAimInputWidth = 119

type SpatialAimFile struct {
	Version string       `json:"version"`
	Layers  []DenseLayer `json:"layers"`
}

func validateSpatialAim(f *SpatialAimFile) error {
	if f == nil {
		return nil
	}
	if f.Version != SpatialAimVersion || len(f.Layers) != 3 {
		return fmt.Errorf("invalid spatial aim header")
	}
	width := SpatialAimInputWidth
	for i, layer := range f.Layers {
		if len(layer.Weight) != len(layer.Bias) || len(layer.Bias) == 0 || len(layer.Bias) > MaxHiddenWidth || i == 2 && len(layer.Bias) != 4 {
			return fmt.Errorf("invalid spatial aim dimensions")
		}
		for row, weights := range layer.Weight {
			if len(weights) != width {
				return fmt.Errorf("invalid spatial aim input")
			}
			for _, value := range append(append([]float64(nil), weights...), layer.Bias[row]) {
				if math.IsNaN(value) || math.IsInf(value, 0) || math.Abs(value) > 1e4 {
					return fmt.Errorf("invalid spatial aim parameter")
				}
			}
		}
		width = len(layer.Bias)
	}
	return nil
}

// All inputs are current observed features and the actor's planned mean move.
// No future displacement, target ID, native outcomes or tactical teacher enters.
func spatialAimInputs(features, raw []float64, slot int) []float64 {
	e := features[73+12*slot : 85+12*slot]
	typed := features[466+40*slot : 506+40*slot]
	x := append([]float64(nil), features[:20]...)
	x = append(x, features[810:845]...)
	x = append(x, math.Tanh(raw[0]), math.Tanh(raw[1]))
	x = append(x, e...)
	x = append(x, typed...)
	x = append(x, features[426+5*slot:431+5*slot]...)
	distance := math.Max(1, e[1]*512)
	px, py, pz := e[2]*512, e[3]*512, e[4]*512
	vx := (e[6] - features[10]*features[7] - features[11]*features[6]) * 400
	vy := (e[7] + features[10]*features[6] - features[11]*features[7]) * 400
	vz := (e[8] - features[12]) * 400
	closing := e[5] * (px*vx + py*vy + pz*vz) / math.Max(1, math.Sqrt(px*px+py*py+pz*pz)) / 400
	angular := e[5] * (px*vy - py*vx) / math.Max(1, px*px+py*py) / 4
	x = append(x, 1/(1+distance/64), math.Atan2(typed[37]*64, distance)/math.Pi, math.Atan2((typed[39]-typed[38])*32, distance)/math.Pi, math.Max(-4, math.Min(4, closing)), math.Max(-4, math.Min(4, angular)))
	return x
}

func (p *PPO) spatialRaw(o Observation, raw []float64) ([]float64, error) {
	if p.file.SpatialAim == nil {
		return raw, nil
	}
	if len(raw) != PrecisionOutputWidth {
		return nil, fmt.Errorf("invalid spatial actor output")
	}
	features, err := FeaturesForVersion(o, TargetFeatureVersion)
	if err != nil {
		return nil, err
	}
	result := append([]float64(nil), raw...)
	for slot := 0; slot < 8; slot++ {
		x := spatialAimInputs(features, raw, slot)
		for index, layer := range p.file.SpatialAim.Layers {
			y := make([]float64, len(layer.Bias))
			for r, weights := range layer.Weight {
				y[r] = layer.Bias[r]
				for c, w := range weights {
					y[r] += w * x[c]
				}
				if index < 2 {
					y[r] = math.Max(0, y[r])
				}
				if math.IsNaN(y[r]) || math.IsInf(y[r], 0) {
					return nil, fmt.Errorf("nonfinite spatial aim output")
				}
			}
			x = y
		}
		for axis := 0; axis < 2; axis++ {
			result[PrecisionFineOffset+2*(slot+1)+axis] += x[axis]
			result[PrecisionModeOffset+2*(slot+1)+axis] += x[axis+2]
		}
	}
	return result, nil
}
