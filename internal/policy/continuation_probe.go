package policy

import (
	"fmt"
	"math"
)

const DiagnosticContinuationVersion = "combat_policy_continuation_rng_fork_v1"

type DiagnosticContinuationFile struct {
	Version string `json:"version"`
	Frame   int    `json:"fork_frame"`
	Seed    int64  `json:"seed"`
}

// SamplingSeed remains the initial episode seed. This diagnostic receipt names
// the effective post-fork stream explicitly and preserves its reused draws.
type DiagnosticContinuationSample struct {
	Version       string    `json:"version"`
	Frame         int       `json:"fork_frame"`
	Seed          int64     `json:"seed"`
	Reset         bool      `json:"reset"`
	GaussianNoise []float64 `json:"gaussian_noise"`
	FireUniform   float64   `json:"fire_uniform"`
}

func validateDiagnosticContinuation(f PPOFile) error {
	p := f.DiagnosticContinuation
	if p == nil {
		return nil
	}
	if p.Version != DiagnosticContinuationVersion || p.Frame <= 1 || p.Frame > math.MaxInt32 || p.Seed < 0 || p.Seed > math.MaxInt32 ||
		f.Deterministic || f.Features != EquippedWeaponFeatureVersion || f.ActionDistribution != JointAimCandidateVersion || f.AimCandidateHead == nil ||
		f.DiagnosticAimProbe != nil || f.ExecutionMode != "" || f.GeometryAimRanker != nil || f.GeometryAimExplore != nil {
		return fmt.Errorf("invalid diagnostic continuation contract")
	}
	if m := f.DiagnosticMovementProbe; m != nil && m.Frame+1 != p.Frame {
		return fmt.Errorf("continuation must fork immediately after movement probe")
	}
	return nil
}

func (p *PPO) continuationSamplingSeed(o Observation) int64 {
	if f := p.file.DiagnosticContinuation; f != nil && o.Identity.Life == 1 && o.Identity.Frame >= f.Frame {
		return f.Seed
	}
	return p.file.SamplingSeed
}

// Call once per real Decide, before any random draw; never during Review/export.
func (p *PPO) beginContinuation(o Observation) error {
	f := p.file.DiagnosticContinuation
	if f == nil {
		return nil
	}
	if o.Identity.Life != 1 {
		return fmt.Errorf("continuation probe requires first life")
	}
	if o.Identity.Frame < f.Frame {
		if p.continuationStarted {
			return fmt.Errorf("continuation probe frame moved backwards")
		}
		return nil
	}
	if !p.continuationStarted {
		if o.Identity.Frame != f.Frame {
			return fmt.Errorf("continuation fork frame was skipped")
		}
		p.rng.Seed(f.Seed)
		p.continuationStarted = true
	}
	return nil
}

func (p *PPO) validateContinuationSample(o Observation, s Sample) error {
	f, r := p.file.DiagnosticContinuation, s.ContinuationProbe
	if f != nil && o.Identity.Life != 1 {
		return fmt.Errorf("continuation receipt requires first life")
	}
	if f == nil || o.Identity.Frame < f.Frame {
		if r != nil {
			return fmt.Errorf("unexpected continuation receipt")
		}
		return nil
	}
	if o.Identity.Life != 1 || r == nil || r.Version != DiagnosticContinuationVersion || r.Frame != f.Frame || r.Seed != f.Seed ||
		r.Reset != (o.Identity.Frame == f.Frame) || len(r.GaussianNoise) != 4 || math.IsNaN(r.FireUniform) || r.FireUniform < 0 || r.FireUniform >= 1 {
		return fmt.Errorf("invalid continuation receipt")
	}
	for _, v := range r.GaussianNoise {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return fmt.Errorf("nonfinite continuation noise")
		}
	}
	return nil
}
