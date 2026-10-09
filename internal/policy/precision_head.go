package policy

import (
	"fmt"
	"math"
)

const PrecisionHeadVersion = "combat_target_coarse_fine_aim_v1"
const PrecisionOutputWidth = 81
const PrecisionFineOffset = 45
const PrecisionModeOffset = 63
const PrecisionFineDegrees = 15.

func precisionLogProbabilities(raw []float64, target int) ([2]float64, error) {
	if len(raw) != PrecisionOutputWidth || target < 0 || target > 8 {
		return [2]float64{}, fmt.Errorf("invalid precision action")
	}
	a, b := raw[PrecisionModeOffset+2*target], raw[PrecisionModeOffset+2*target+1]
	if math.IsNaN(a) || math.IsInf(a, 0) || math.IsNaN(b) || math.IsInf(b, 0) {
		return [2]float64{}, fmt.Errorf("nonfinite precision logits")
	}
	m := math.Max(a, b)
	z := math.Log(math.Exp(a-m) + math.Exp(b-m))
	return [2]float64{a - m - z, b - m - z}, nil
}

func precisionMeans(raw []float64, target, mode int) ([4]float64, error) {
	if len(raw) != PrecisionOutputWidth || mode < 0 || mode > 1 {
		return [4]float64{}, fmt.Errorf("invalid precision mode")
	}
	means, err := targetMeans(raw[:TargetOutputWidth], target)
	if err != nil {
		return means, err
	}
	if mode == 1 {
		means[2] = raw[PrecisionFineOffset+2*target]
		means[3] = raw[PrecisionFineOffset+2*target+1]
	}
	return means, nil
}
