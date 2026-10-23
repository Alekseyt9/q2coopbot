package policy

import (
	"fmt"
	"math"
)

const DiagnosticMovementProbeVersion = "combat_one_frame_movement_mean_shift_v1"

// Diagnostic acquisition only. Shift the two Gaussian movement means on one
// frame; retain std, draw order and the ordinary motion-conditioned aim.
type DiagnosticMovementProbeFile struct {
	Version   string    `json:"version"`
	Frame     int       `json:"frame"`
	MeanShift []float64 `json:"mean_shift"`
}

type DiagnosticMovementSample struct {
	Version       string    `json:"version"`
	Frame         int       `json:"frame"`
	MeanShift     []float64 `json:"mean_shift"`
	GaussianNoise []float64 `json:"gaussian_noise"`
	FireUniform   float64   `json:"fire_uniform"`
}

func (p *PPO) validateMovementSample(frame int, sample Sample) error {
	probe, receipt := p.file.DiagnosticMovementProbe, sample.MovementProbe
	if probe == nil || frame != probe.Frame {
		if receipt != nil {
			return fmt.Errorf("unexpected movement probe receipt")
		}
		return nil
	}
	if receipt == nil || receipt.Version != DiagnosticMovementProbeVersion || receipt.Frame != frame ||
		len(receipt.MeanShift) != 2 || len(receipt.GaussianNoise) != 4 ||
		math.IsNaN(receipt.FireUniform) || receipt.FireUniform < 0 || receipt.FireUniform >= 1 {
		return fmt.Errorf("invalid movement probe draw receipt")
	}
	for i, v := range receipt.MeanShift {
		if v != probe.MeanShift[i] {
			return fmt.Errorf("movement probe shift differs")
		}
	}
	for _, v := range receipt.GaussianNoise {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return fmt.Errorf("nonfinite movement draw")
		}
	}
	return nil
}

func validateDiagnosticMovementProbe(f PPOFile) error {
	p := f.DiagnosticMovementProbe
	if p == nil {
		return nil
	}
	if p.Version != DiagnosticMovementProbeVersion || p.Frame <= 0 || p.Frame > math.MaxInt32 || len(p.MeanShift) != 2 ||
		f.Deterministic || f.Features != EquippedWeaponFeatureVersion || f.ActionDistribution != JointAimCandidateVersion ||
		f.AimCandidateHead == nil || f.DiagnosticAimProbe != nil || f.ExecutionMode != "" || f.GeometryAimRanker != nil || f.GeometryAimExplore != nil {
		return fmt.Errorf("invalid diagnostic movement probe contract")
	}
	for _, v := range p.MeanShift {
		if math.IsNaN(v) || math.IsInf(v, 0) || math.Abs(v) > 2 {
			return fmt.Errorf("invalid diagnostic movement mean shift")
		}
	}
	return nil
}

func (p *PPO) movementProbeMeans(frame int, means [4]float64) [4]float64 {
	probe := p.file.DiagnosticMovementProbe
	if probe == nil || probe.Frame != frame {
		return means
	}
	for i, shift := range probe.MeanShift {
		if shift != 0 { // Preserve exact zero-shift control, including signed zero.
			means[i] += shift
		}
	}
	return means
}
