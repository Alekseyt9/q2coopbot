package policy

import (
	"fmt"
	"math"
)

const WideAimDistribution = "combat_joint_aim_wide_v2"
const WideAimVersion = "combat_observed_mg_wide_choice_v1"
const WideAimSampler = "frame_hash_wide_bernoulli_v1"
const WideAimInputWidth = 187

// A learned option following the ordinary Gaussian and 25-way correction.
// Disabled models preserve the legacy event and RNG stream exactly.
type WideAimFile struct {
	Version string       `json:"version"`
	Sampler string       `json:"sampler"`
	Enabled bool         `json:"enabled"`
	Layers  []DenseLayer `json:"layers"`
}

type WideAimSample struct {
	Version        string     `json:"version"`
	SourceLatent   [4]float64 `json:"source_latent"`
	Anchor         [2]float64 `json:"anchor_degrees"`
	Choice         int        `json:"choice"`
	Uniform        float64    `json:"uniform"`
	FireUniform    float64    `json:"fire_uniform"`
	Logit          float64    `json:"logit"`
	LogProbability float64    `json:"log_probability"`
}

func validateWideAim(f PPOFile) error {
	h := f.WideAim
	if h == nil {
		if f.ActionDistribution == WideAimDistribution {
			return fmt.Errorf("wide event requires head")
		}
		return nil
	}
	if f.ActionDistribution != WideAimDistribution || h.Version != WideAimVersion || h.Sampler != WideAimSampler ||
		f.Deterministic || f.ExecutionMode != "" || f.CombatExecution != StockWeaponsFireCombatExecution ||
		!equippedPrefixFeatures(f.Features) || f.TargetHead != TargetHeadVersion || f.AimModeHead != PrecisionHeadVersion ||
		f.ConditionalFire == nil || f.AimCandidateHead == nil || len(h.Layers) != 3 ||
		f.DiagnosticAimProbe != nil || f.DiagnosticAnchorProbe != nil || f.DiagnosticMovementProbe != nil ||
		f.DiagnosticWeaponProbe != nil || f.DiagnosticContinuation != nil || f.GeometryAimRanker != nil || f.GeometryAimExplore != nil {
		return fmt.Errorf("invalid learned wide aim contract")
	}
	width := WideAimInputWidth
	for i, output := range []int{32, 16, 1} {
		l := h.Layers[i]
		if len(l.Weight) != output || len(l.Bias) != output {
			return fmt.Errorf("invalid wide aim dimensions")
		}
		for r, row := range l.Weight {
			if len(row) != width {
				return fmt.Errorf("invalid wide aim input width")
			}
			for _, v := range append(append([]float64(nil), row...), l.Bias[r]) {
				if math.IsNaN(v) || math.IsInf(v, 0) || math.Abs(v) > 1e4 {
					return fmt.Errorf("invalid wide aim weight")
				}
			}
		}
		width = output
	}
	return nil
}

// This uses only current observed geometry/recoil, never the future native pose.
func observedWideAim(o Observation, s Sample) ([2]float64, bool, error) {
	var anchor [2]float64
	if o.Health <= 0 || equippedWeaponIndex(o.Weapon) != 4 || s.AimMode != 0 || o.KickAngles == nil || s.Target <= 0 || s.Target > len(TargetEnemies(o)) {
		return anchor, false, nil
	}
	e := TargetEnemies(o)[s.Target-1]
	e.Relative[2] += 8 // centered MG muzzle = eye - 8
	d, ok, err := ObservedAimDirection(o, e)
	if err != nil || !ok {
		return anchor, false, err
	}
	for _, v := range *o.KickAngles {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return anchor, false, fmt.Errorf("invalid wide aim kick")
		}
	}
	wrap := func(v float64) float64 { return v - 360*math.Floor((v+180)/360) }
	anchor = [2]float64{wrap(math.Atan2(d[1], d[0])*180/math.Pi - degrees(o.ViewAngles[1]) - (*o.KickAngles)[1]),
		wrap(-math.Atan2(d[2], math.Hypot(d[0], d[1]))*180/math.Pi - degrees(o.ViewAngles[0]) - (*o.KickAngles)[0])}
	return anchor, true, nil
}

func WideAimUniform(seed int64, frame, life int) float64 {
	u, _ := JointAimUniforms(seed^int64(0x6a09e667f3bcc909), frame, life)
	return u
}

func applyWideChoice(source [4]float64, anchor [2]float64, choice int) ([4]float64, error) {
	for _, v := range source {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return source, fmt.Errorf("invalid wide source")
		}
	}
	if choice < 0 || choice > 1 {
		return source, fmt.Errorf("invalid wide choice")
	}
	for _, v := range anchor {
		if math.IsNaN(v) || math.IsInf(v, 0) || v < -180 || v >= 180 {
			return source, fmt.Errorf("invalid wide anchor")
		}
	}
	if choice == 1 {
		for i, v := range anchor {
			source[2+i] = math.Atanh(math.Max(-1+1e-6, math.Min(1-1e-6, v/180)))
		}
	}
	return source, nil
}

func wideLogProbability(logit float64, choice int) float64 {
	if choice == 1 {
		return -softplus(-logit)
	}
	return -softplus(logit)
}

// Metadata and geometry only. Return the legacy post-25 event for its existing
// verifier, without losing the original Gaussian sample or changing movement.
func (p *PPO) legacyWideSample(o Observation, s Sample) (Sample, error) {
	r, h := s.WideAim, p.file.WideAim
	if h == nil || !h.Enabled {
		if r != nil {
			return s, fmt.Errorf("wide aim event on disabled/legacy model")
		}
		return s, nil
	}
	anchor, eligible, err := observedWideAim(o, s)
	if err != nil {
		return s, err
	}
	if !eligible {
		if r != nil {
			return s, fmt.Errorf("unexpected wide aim event")
		}
		return s, nil
	}
	if r == nil || r.Version != WideAimVersion || !validProbeUniform(r.FireUniform) || math.IsNaN(r.Logit) || math.IsInf(r.Logit, 0) || math.Abs(r.Logit) > 20 ||
		math.IsNaN(r.LogProbability) || math.IsInf(r.LogProbability, 0) || r.Anchor != anchor ||
		r.Uniform != WideAimUniform(p.continuationSamplingSeed(o), o.Identity.Frame, o.Identity.Life) ||
		(r.Uniform < math.Exp(-softplus(-r.Logit))) != (r.Choice == 1) ||
		math.Abs(r.LogProbability-wideLogProbability(r.Logit, r.Choice)) > 1e-9 {
		return s, fmt.Errorf("invalid wide aim event receipt")
	}
	final, err := applyWideChoice(r.SourceLatent, anchor, r.Choice)
	if err != nil {
		return s, err
	}
	for i, v := range final {
		if math.IsNaN(s.Latent[i]) || math.IsInf(s.Latent[i], 0) || math.Abs(v-s.Latent[i]) > 1e-9 {
			return s, fmt.Errorf("wide aim transform differs")
		}
	}
	s.Latent = r.SourceLatent
	s.WideAim = nil
	return s, nil
}

func (p *PPO) wideAimLogit(o Observation, s Sample, anchor [2]float64) (float64, error) {
	f, err := FeaturesForVersion(o, EquippedWeaponFeatureVersion)
	if err != nil {
		return 0, err
	}
	x, err := GeometryActionInputs(f, s)
	if err != nil {
		return 0, err
	}
	x = append(x, anchor[0]/180, anchor[1]/180)
	for i, l := range p.file.WideAim.Layers {
		y := make([]float64, len(l.Bias))
		for r, row := range l.Weight {
			y[r] = l.Bias[r]
			for c, w := range row {
				y[r] += w * x[c]
			}
			if i < 2 {
				y[r] = math.Max(0, y[r])
			}
			if math.IsNaN(y[r]) || math.IsInf(y[r], 0) {
				return 0, fmt.Errorf("nonfinite wide aim logit")
			}
		}
		x = y
	}
	return math.Max(-20, math.Min(20, x[0])), nil
}

func (p *PPO) selectWideAim(o Observation, s *Sample, fireUniform float64) error {
	if p.file.WideAim == nil || !p.file.WideAim.Enabled {
		return nil
	}
	a, active, err := observedWideAim(o, *s)
	if err != nil || !active {
		return err
	}
	l, err := p.wideAimLogit(o, *s, a)
	if err != nil {
		return err
	}
	u := WideAimUniform(p.continuationSamplingSeed(o), o.Identity.Frame, o.Identity.Life)
	choice := 0
	if u < math.Exp(-softplus(-l)) {
		choice = 1
	}
	s.WideAim = &WideAimSample{Version: WideAimVersion, SourceLatent: s.Latent, Anchor: a, Choice: choice, Uniform: u, FireUniform: fireUniform, Logit: l, LogProbability: wideLogProbability(l, choice)}
	s.Latent, err = applyWideChoice(s.Latent, a, choice)
	return err
}

func (p *PPO) reviewWideAim(o Observation, s Sample) (float64, error) {
	base, err := p.legacyWideSample(o, s)
	if err != nil {
		return 0, err
	}
	if s.WideAim == nil {
		return 0, nil
	}
	l, err := p.wideAimLogit(o, base, s.WideAim.Anchor)
	if err != nil {
		return 0, err
	}
	if math.Abs(l-s.WideAim.Logit) > 1e-9 {
		return 0, fmt.Errorf("wide aim neural receipt differs")
	}
	return wideLogProbability(l, s.WideAim.Choice), nil
}
