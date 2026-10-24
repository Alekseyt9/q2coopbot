package policy

import (
	"fmt"
	"math"
)

const DiagnosticWeaponProbeVersion = "combat_one_frame_weapon_choice_v1"

// Acquisition only: identity retains the source CDF, force replaces it once.
type DiagnosticWeaponProbeFile struct {
	Version                string `json:"version"`
	Frame                  int    `json:"frame"`
	Mode                   string `json:"mode"`
	ExpectedEquippedWeapon int    `json:"expected_equipped_weapon"`
	ForcedWeapon           int    `json:"forced_weapon"`
}

func (p *DiagnosticWeaponProbeFile) active(frame int) bool { return p != nil && p.Frame == frame }

type DiagnosticWeaponSample struct {
	Version              string    `json:"version"`
	Frame                int       `json:"frame"`
	Mode                 string    `json:"mode"`
	SourceProbabilities  []float64 `json:"source_probabilities"`
	AppliedProbabilities []float64 `json:"applied_probabilities"`
	WeaponUniform        float64   `json:"weapon_uniform"`
	FireUniform          float64   `json:"fire_uniform"`
	GaussianNoise        []float64 `json:"gaussian_noise"`
}

func validateDiagnosticWeaponProbe(f PPOFile) error {
	p := f.DiagnosticWeaponProbe
	if p == nil {
		return nil
	}
	if p.Version != DiagnosticWeaponProbeVersion || p.Frame <= 0 || p.Frame > math.MaxInt32 ||
		(p.Mode != "identity" && p.Mode != "force") || p.ExpectedEquippedWeapon != 1 || p.ForcedWeapon != 2 ||
		f.Deterministic || !equippedPrefixFeatures(f.Features) || f.WeaponHead != WeaponHeadVersion || f.TargetHead != TargetHeadVersion ||
		f.ActionDistribution != JointAimCandidateVersion || f.AimCandidateHead == nil || f.ConditionalFire == nil ||
		f.DiagnosticAimProbe != nil || f.DiagnosticMovementProbe != nil || f.DiagnosticContinuation != nil ||
		f.ExecutionMode != "" || f.GeometryAimRanker != nil || f.GeometryAimExplore != nil {
		return fmt.Errorf("invalid diagnostic weapon probe contract")
	}
	return nil
}

func (p *PPO) weaponProbeGuard(o Observation, target int) error {
	f := p.file.DiagnosticWeaponProbe
	if o.Identity.Life != 1 || equippedWeaponIndex(o.Weapon) != f.ExpectedEquippedWeapon ||
		target <= 0 || target > len(TargetEnemies(o)) || o.InventoryAgeFrames == nil ||
		*o.InventoryAgeFrames < 0 || *o.InventoryAgeFrames > 20 {
		return fmt.Errorf("diagnostic weapon root observation differs")
	}
	mask, err := WeaponAvailability(o)
	if err != nil {
		return err
	}
	if !mask[f.ForcedWeapon] {
		return fmt.Errorf("diagnostic forced weapon unavailable")
	}
	return nil
}

func (p *PPO) weaponProbeProbabilities(o Observation, target int, source []float64) ([]float64, error) {
	f := p.file.DiagnosticWeaponProbe
	if !f.active(o.Identity.Frame) {
		return source, nil
	}
	if err := p.weaponProbeGuard(o, target); err != nil {
		return nil, err
	}
	if f.Mode == "identity" {
		return source, nil
	}
	applied := make([]float64, len(source))
	applied[f.ForcedWeapon] = 1
	return applied, nil
}

func validProbeUniform(v float64) bool { return !math.IsNaN(v) && v >= 0 && v < 1 }

func (p *PPO) reviewWeaponProbe(o Observation, s Sample, logs []float64) ([]float64, error) {
	f, r := p.file.DiagnosticWeaponProbe, s.WeaponProbe
	if !f.active(o.Identity.Frame) {
		if r != nil {
			return nil, fmt.Errorf("unexpected weapon probe receipt")
		}
		return logs, nil
	}
	if r == nil || r.Version != f.Version || r.Frame != f.Frame || r.Mode != f.Mode ||
		len(r.SourceProbabilities) != len(weaponNames) || len(r.AppliedProbabilities) != len(weaponNames) ||
		len(r.GaussianNoise) != 4 || !validProbeUniform(r.WeaponUniform) || !validProbeUniform(r.FireUniform) {
		return nil, fmt.Errorf("invalid weapon probe receipt")
	}
	source := make([]float64, len(logs))
	for i, v := range logs {
		source[i] = math.Exp(v)
	}
	applied, err := p.weaponProbeProbabilities(o, s.Target, source)
	if err != nil {
		return nil, err
	}
	for i := range source {
		if r.SourceProbabilities[i] != source[i] || r.AppliedProbabilities[i] != applied[i] {
			return nil, fmt.Errorf("weapon probe distribution receipt differs")
		}
	}
	for _, v := range r.GaussianNoise {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return nil, fmt.Errorf("invalid weapon probe noise")
		}
	}
	u, selected := r.WeaponUniform, 0
	for i, probability := range applied {
		if probability == 0 {
			continue
		}
		selected = i
		u -= probability
		if u <= 0 {
			break
		}
	}
	if selected != s.Weapon {
		return nil, fmt.Errorf("weapon probe sampled category differs")
	}
	if f.Mode == "force" {
		for i := range logs {
			logs[i] = math.Inf(-1)
		}
		logs[f.ForcedWeapon] = 0
	}
	return logs, nil
}
