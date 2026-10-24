package policy

import (
	"fmt"
	"math"
)

const TargetSamplingVersion = "combat_target_temperature_v1"

type TargetSamplingFile struct {
	Version     string  `json:"version"`
	Temperature float64 `json:"temperature"`
}

func validateTargetSampling(f PPOFile) error {
	h := f.TargetSampling
	if h == nil {
		return nil
	}
	if h.Version != TargetSamplingVersion || math.IsNaN(h.Temperature) || math.IsInf(h.Temperature, 0) ||
		h.Temperature < .25 || h.Temperature > 2 || f.Features != EquippedWeaponFeatureVersion ||
		f.TargetHead != TargetHeadVersion || f.ActionDistribution != JointAimCandidateVersion || f.AimCandidateHead == nil ||
		f.EntityAttention != nil || f.MovementInertia != nil || f.MemoryRecovery != nil ||
		f.DiagnosticAimProbe != nil || f.DiagnosticMovementProbe != nil || f.DiagnosticContinuation != nil || f.ExecutionMode != "" {
		return fmt.Errorf("invalid target sampling contract")
	}
	return nil
}

// Scale only the categorical factor. Keep the actor outputs unchanged for all
// conditional movement/aim/fire branches, and consume the usual target draw.
func (p *PPO) targetSamplingLogProbabilities(o Observation, logits []float64) ([]float64, error) {
	h := p.file.TargetSampling
	if h == nil || h.Temperature == 1 {
		return targetLogProbabilities(o, logits)
	}
	scaled := make([]float64, len(logits))
	for i, value := range logits {
		scaled[i] = value / h.Temperature
	}
	return targetLogProbabilities(o, scaled)
}
