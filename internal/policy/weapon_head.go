package policy

import (
	"fmt"
	"math"
)

const WeaponHeadVersion = "combat_masked_weapon_v1"

// Mask before normalization: unavailable actions have exactly zero probability.
func weaponProbabilities(o Observation, logits []float64) ([]float64, error) {
	logProbabilities, err := weaponLogProbabilities(o, logits)
	if err != nil {
		return nil, err
	}
	for i := range logProbabilities {
		logProbabilities[i] = math.Exp(logProbabilities[i])
	}
	return logProbabilities, nil
}

func weaponLogProbabilities(o Observation, logits []float64) ([]float64, error) {
	if len(logits) != len(weaponNames) {
		return nil, fmt.Errorf("invalid weapon logits width")
	}
	mask, err := WeaponAvailability(o)
	if err != nil {
		return nil, err
	}
	maximum := -math.MaxFloat64
	for i, value := range logits {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return nil, fmt.Errorf("nonfinite weapon logit")
		}
		if mask[i] {
			maximum = math.Max(maximum, value)
		}
	}
	probabilities := make([]float64, len(logits))
	sum := 0.0
	for i, value := range logits {
		if mask[i] {
			probabilities[i] = math.Exp(value - maximum)
			sum += probabilities[i]
		}
	}
	for i := range probabilities {
		probabilities[i] = math.Inf(-1)
		if mask[i] {
			probabilities[i] = logits[i] - maximum - math.Log(sum)
		}
	}
	return probabilities, nil
}
