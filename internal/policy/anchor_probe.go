package policy

import (
	"fmt"
	"math"
)

const DiagnosticAnchorProbeVersion = "combat_one_frame_observed_mg_anchor_v1"

// This changes actuation, not the sampled policy event. Its likelihood remains
// the SOURCE event likelihood and must never enter ordinary PPO training.
type DiagnosticAnchorProbeFile struct {
	Version string `json:"version"`
	Frame   int    `json:"frame"`
	Mode    string `json:"mode"`
}

type DiagnosticAnchorSample struct {
	Version      string     `json:"version"`
	Frame        int        `json:"frame"`
	Mode         string     `json:"mode"`
	TargetEntity int        `json:"target_entity"`
	Source       [2]float64 `json:"source_degrees"`
	Returned     [2]float64 `json:"returned_degrees"`
}

func validateDiagnosticAnchorProbe(f PPOFile) error {
	p := f.DiagnosticAnchorProbe
	if p == nil {
		return nil
	}
	if p.Version != DiagnosticAnchorProbeVersion || p.Frame <= 0 || p.Frame > math.MaxInt32 ||
		(p.Mode != "identity" && p.Mode != "observed_mg") || f.Deterministic ||
		!equippedPrefixFeatures(f.Features) || f.TargetHead != TargetHeadVersion || f.WeaponHead != WeaponHeadVersion ||
		f.AimModeHead != PrecisionHeadVersion || f.ActionDistribution != JointAimCandidateVersion || f.AimCandidateHead == nil ||
		f.DiagnosticAimProbe != nil || f.DiagnosticMovementProbe != nil || f.DiagnosticWeaponProbe != nil || f.DiagnosticContinuation != nil ||
		f.ExecutionMode != "" || f.GeometryAimRanker != nil || f.GeometryAimExplore != nil {
		return fmt.Errorf("invalid diagnostic observed anchor contract")
	}
	return nil
}

func (p *PPO) IsDiagnosticAnchorProbe() bool { return p.file.DiagnosticAnchorProbe != nil }

func (p *PPO) anchorReceipt(o Observation, s Sample) (*DiagnosticAnchorSample, error) {
	f := p.file.DiagnosticAnchorProbe
	if f == nil || f.Frame != o.Identity.Frame {
		return nil, nil
	}
	if o.Identity.Life != 1 || o.Health <= 0 || equippedWeaponIndex(o.Weapon) != 4 || s.AimMode != 0 ||
		s.Target <= 0 || s.Target > len(TargetEnemies(o)) || o.KickAngles == nil {
		return nil, fmt.Errorf("diagnostic anchor root observation differs")
	}
	e := TargetEnemies(o)[s.Target-1]
	// CENTER_HANDED MG muzzle is eight units below the observed eye.
	e.Relative[2] += 8
	r, known, err := ObservedAimDirection(o, e)
	if err != nil || !known || e.ID <= 0 {
		return nil, fmt.Errorf("diagnostic anchor needs observed visible bbox")
	}
	for _, x := range *o.KickAngles {
		if math.IsNaN(x) || math.IsInf(x, 0) {
			return nil, fmt.Errorf("nonfinite anchor recoil")
		}
	}
	wrap := func(v float64) float64 { return v - 360*math.Floor((v+180)/360) }
	anchor := [2]float64{
		wrap(math.Atan2(r[1], r[0])*180/math.Pi - degrees(o.ViewAngles[1]) - (*o.KickAngles)[1]),
		wrap(-math.Atan2(r[2], math.Hypot(r[0], r[1]))*180/math.Pi - degrees(o.ViewAngles[0]) - (*o.KickAngles)[0]),
	}
	source := [2]float64{180 * math.Tanh(s.Latent[2]), 180 * math.Tanh(s.Latent[3])}
	for _, x := range source {
		if math.IsNaN(x) || math.IsInf(x, 0) {
			return nil, fmt.Errorf("nonfinite source aim")
		}
	}
	rc := &DiagnosticAnchorSample{Version: f.Version, Frame: f.Frame, Mode: f.Mode, TargetEntity: e.ID, Source: source, Returned: source}
	if f.Mode == "observed_mg" {
		rc.Returned = anchor
	}
	return rc, nil
}

// Metadata/geometry only; it consumes no randomness and runs no network.
func (p *PPO) applyAnchorReceipt(o Observation, s Sample, a Action) (Action, error) {
	expected, err := p.anchorReceipt(o, s)
	if err != nil {
		return Action{}, err
	}
	if expected == nil {
		if s.AnchorProbe != nil {
			return Action{}, fmt.Errorf("unexpected anchor receipt")
		}
		return a, nil
	}
	if s.AnchorProbe == nil || *s.AnchorProbe != *expected || a.TargetEntity != expected.TargetEntity ||
		a.YawDelta != expected.Source[0] || a.PitchDelta != expected.Source[1] {
		return Action{}, fmt.Errorf("anchor receipt differs from observed geometry/source sample")
	}
	a.YawDelta, a.PitchDelta = expected.Returned[0], expected.Returned[1]
	return a, nil
}
