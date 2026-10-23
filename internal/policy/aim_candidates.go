package policy

import (
	"fmt"
	"math"
	"reflect"
)

const JointAimCandidateVersion = "combat_joint_aim_candidates_v1"
const JointAimCandidateWidth = 1359
const GeometryDeltaAimSampler = "frame_hash_geometry_delta31_v1"
const GeometryDeltaAimKind = "geometry_candidate_delta31_mlp_v1"
const DiagnosticAimProbeVersion = "combat_joint_aim_one_frame_probe_v1"
const DiagnosticPolicyAimProbeVersion = "combat_joint_aim_one_frame_policy_probe_v2"

// Diagnostic acquisition only. V1 requires a constant head; V2 preserves the
// ordinary head outside one declared frame. Actor RNG is unchanged.
type DiagnosticAimProbeFile struct {
	Version string    `json:"version"`
	Frame   int       `json:"frame"`
	Logits  []float64 `json:"logits"`
}

func validateProbeLogits(logits []float64) bool {
	if len(logits) != 25 {
		return false
	}
	low, high := math.Inf(1), math.Inf(-1)
	for _, v := range logits {
		if math.IsNaN(v) || math.IsInf(v, 0) || math.Abs(v) > 100 {
			return false
		}
		low, high = math.Min(low, v), math.Max(high, v)
	}
	return high-low <= 60 // Positive support in both float32 and float64.
}

func validateDiagnosticAimProbe(f PPOFile) error {
	p := f.DiagnosticAimProbe
	if p == nil {
		return nil
	}
	if f.AimCandidateHead == nil || len(f.AimCandidateHead.Layers) != 2 {
		return fmt.Errorf("diagnostic aim probe requires a validated joint head")
	}
	if (p.Version != DiagnosticAimProbeVersion && p.Version != DiagnosticPolicyAimProbeVersion) || p.Frame <= 0 || p.Frame > math.MaxInt32 || !validateProbeLogits(p.Logits) {
		return fmt.Errorf("invalid diagnostic one-frame aim probe")
	}
	if p.Version == DiagnosticPolicyAimProbeVersion {
		if f.AimCandidateHead.Shared != nil {
			for _, v := range p.Logits {
				if math.Abs(v) > 40 {
					return fmt.Errorf("shared probe logit outside live record bounds")
				}
			}
		}
		return nil
	}
	if !validateProbeLogits(f.AimCandidateHead.Layers[1].Bias) {
		return fmt.Errorf("invalid constant diagnostic baseline logits")
	}
	for _, row := range f.AimCandidateHead.Layers[1].Weight {
		for _, w := range row {
			if w != 0 {
				return fmt.Errorf("diagnostic aim probe requires a constant baseline head")
			}
		}
	}
	return nil
}

type AimCandidateFile struct {
	Version   string                `json:"version"`
	Offsets   []float64             `json:"offset_degrees"`
	Sampler   string                `json:"sampler"`
	Layers    []DenseLayer          `json:"layers"`
	Predictor GeometryShotPredictor `json:"predictor"`
	Shared    *SharedAimScorerFile  `json:"shared_scorer,omitempty"`
}

// Shared scorer replaces only category logits; the base actor and sampler stay.
type SharedAimScorerFile struct {
	Kind       string       `json:"kind"`
	Layers     []DenseLayer `json:"layers"`
	Beta       float64      `json:"beta"`
	ScoreClip  float64      `json:"score_clip"`
	BetaBounds []float64    `json:"beta_bounds"`
}

func validateSharedAimScorer(s *SharedAimScorerFile) error {
	if s == nil {
		return nil
	}
	if (s.Kind != "shared_candidate_causal191_v1" && s.Kind != GeometryDeltaAimKind) || s.ScoreClip != 20 || !reflect.DeepEqual(s.BetaBounds, []float64{-20, 20}) || math.IsNaN(s.Beta) || math.IsInf(s.Beta, 0) || math.Abs(s.Beta) > 20 || len(s.Layers) != 2 {
		return fmt.Errorf("invalid shared candidate scorer contract")
	}
	width := 191
	hidden := 16
	if s.Kind == GeometryDeltaAimKind {
		width = 31
		hidden = 8
	}
	for i, output := range []int{hidden, 1} {
		l := s.Layers[i]
		if len(l.Weight) != output || len(l.Bias) != output {
			return fmt.Errorf("invalid shared candidate output width")
		}
		for r, row := range l.Weight {
			if len(row) != width {
				return fmt.Errorf("invalid shared candidate input width")
			}
			for _, v := range append(append([]float64(nil), row...), l.Bias[r]) {
				if math.IsNaN(v) || math.IsInf(v, 0) || math.Abs(v) > 1e6 {
					return fmt.Errorf("invalid shared candidate parameter")
				}
			}
		}
		width = output
	}
	return nil
}

// The probability event includes the original Gaussian draw and correction.
// Final clipped angles cannot recover that original draw unambiguously.
type AimCandidateSample struct {
	SourceLatent   [4]float64 `json:"source_latent"`
	Candidate      int        `json:"candidate"`
	Candidates     int        `json:"candidates"`
	LogProbability float64    `json:"log_probability"`
	Timing         [2]float64 `json:"timing"`
	Logits         []float64  `json:"logits,omitempty"`
}

func validateAimCandidates(f PPOFile) error {
	h := f.AimCandidateHead
	if f.DiagnosticAimProbe != nil && h == nil {
		return fmt.Errorf("diagnostic aim probe requires joint head")
	}
	if h == nil && f.ActionDistribution == "" {
		return nil
	}
	expectedSampler := "frame_hash_categorical_v1"
	if h != nil && h.Shared != nil && h.Shared.Kind == GeometryDeltaAimKind {
		expectedSampler = GeometryDeltaAimSampler
	}
	if h == nil || f.ActionDistribution != JointAimCandidateVersion || h.Version != JointAimCandidateVersion ||
		f.Deterministic || f.ExecutionMode != "" || f.GeometryAimExplore != nil || f.GeometryAimRanker != nil ||
		f.Features != EquippedWeaponFeatureVersion || f.TargetHead != TargetHeadVersion || f.AimModeHead != PrecisionHeadVersion ||
		h.Sampler != expectedSampler || !reflect.DeepEqual(h.Offsets, []float64{0, -5, -2.5, 2.5, 5}) ||
		h.Predictor.GeometryVersion != ActionEffectGeometryVersion || len(h.Layers) != 2 {
		return fmt.Errorf("invalid joint aim candidate contract")
	}
	if err := validateGeometryAimRanker(PPOFile{Features: f.Features, GeometryAimRanker: &GeometryAimRankerFile{Version: GeometryAimRankerVersion, Offsets: h.Offsets, Predictor: h.Predictor}}); err != nil {
		return err
	}
	if err := validateSharedAimScorer(h.Shared); err != nil {
		return err
	}
	if h.Shared != nil && f.DiagnosticAimProbe != nil && f.DiagnosticAimProbe.Version != DiagnosticPolicyAimProbeVersion {
		return fmt.Errorf("shared scorer cannot use constant one-frame probe")
	}
	width := JointAimCandidateWidth
	for i, output := range []int{64, 25} {
		l := h.Layers[i]
		if len(l.Weight) != output || len(l.Bias) != output {
			return fmt.Errorf("invalid candidate output width")
		}
		for r, row := range l.Weight {
			if len(row) != width {
				return fmt.Errorf("invalid candidate input width")
			}
			for _, v := range append(append([]float64(nil), row...), l.Bias[r]) {
				if math.IsNaN(v) || math.IsInf(v, 0) || math.Abs(v) > 1e6 {
					return fmt.Errorf("invalid candidate parameter")
				}
			}
		}
		width = output
	}
	return validateDiagnosticAimProbe(f)
}

// ApplyJointAimOffsets is pure action geometry, also usable by offline packers.
func ApplyJointAimOffsets(source [4]float64, mode, index int) ([4]float64, error) {
	if mode < 0 || mode > 1 || index < 0 || index >= 25 {
		return source, fmt.Errorf("invalid candidate category")
	}
	for _, v := range source {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return source, fmt.Errorf("nonfinite source latent")
		}
	}
	if index == 0 {
		return source, nil
	}
	scale := 180.
	if mode == 1 {
		scale = PrecisionFineDegrees
	}
	offsets := []float64{0, -5, -2.5, 2.5, 5}
	for axis, offset := range []float64{offsets[index/5], offsets[index%5]} {
		ratio := (math.Tanh(source[2+axis])*scale + offset) / scale
		source[2+axis] = math.Atanh(math.Max(-1+1e-6, math.Min(1-1e-6, ratio)))
	}
	return source, nil
}

// JointAimUniforms preserves the source actor RNG stream.
func JointAimUniforms(seed int64, frame, life int) (float64, float64) {
	word := mixAimWord(uint64(seed) ^ uint64(frame)*0x9e3779b97f4a7c15 ^ uint64(life)*0xd1b54a32d192ed03)
	k := mixAimWord(word^0x8cb92baa3f3d8dd7) % 24
	jitter := float64(mixAimWord(word^0x4f1bbcdc6762c5d9)>>11) / 9007199254740992.
	return float64(word>>11) / 9007199254740992., (float64(k) + jitter) / 24
}

func JointAimCandidateInputs(features, base, hints []float64, timing [2]float64) ([]float64, error) {
	if len(features) != 1147 || len(base) != 185 || len(hints) != 25 {
		return nil, fmt.Errorf("candidate input widths differ")
	}
	x := make([]float64, 0, JointAimCandidateWidth)
	x = append(x, features...)
	x = append(x, base...)
	x = append(x, hints...)
	x = append(x, timing[:]...)
	for _, v := range x {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return nil, fmt.Errorf("nonfinite candidate input")
		}
	}
	return x, nil
}

func jointAimActive(c []float64) bool {
	return len(c) == 185 && c[77] > 0 && c[151] == 0 && c[99] > 0 && c[147] > 0
}

// Metadata and transform validation only: no network evaluation here.
func (p *PPO) jointAimSample(o Observation, s Sample) (Sample, []float64, bool, error) {
	if p.file.AimCandidateHead == nil {
		if s.AimCandidate != nil {
			return s, nil, false, fmt.Errorf("candidate sample on legacy policy")
		}
		return s, nil, false, nil
	}
	if s.AimRanking != nil || s.AimExploration != nil {
		return s, nil, false, fmt.Errorf("inference-only annotation on joint sample")
	}
	base := s
	if s.AimCandidate != nil {
		base.Latent = s.AimCandidate.SourceLatent
	}
	f, err := FeaturesForVersion(o, EquippedWeaponFeatureVersion)
	if err != nil {
		return base, nil, false, err
	}
	c, err := GeometryActionInputs(f, base)
	if err != nil {
		return base, nil, false, err
	}
	active := jointAimActive(c)
	if !active {
		if s.AimCandidate != nil {
			return base, nil, false, fmt.Errorf("candidate annotation outside eligible context")
		}
		return base, c, false, nil
	}
	r := s.AimCandidate
	if r == nil || r.Candidates != 25 || math.IsNaN(r.LogProbability) || math.IsInf(r.LogProbability, 0) {
		return base, nil, false, fmt.Errorf("missing/invalid candidate event")
	}
	if p.file.AimCandidateHead.Shared != nil {
		if len(r.Logits) != 25 {
			return base, nil, false, fmt.Errorf("missing shared candidate live logits")
		}
		for _, v := range r.Logits {
			if math.IsNaN(v) || math.IsInf(v, 0) || math.Abs(v) > 40 {
				return base, nil, false, fmt.Errorf("invalid shared candidate live logits")
			}
		}
	} else if len(r.Logits) != 0 {
		return base, nil, false, fmt.Errorf("unexpected shared candidate live logits")
	}
	final, err := ApplyJointAimOffsets(base.Latent, s.AimMode, r.Candidate)
	if err != nil {
		return base, nil, false, err
	}
	for i, v := range final {
		if math.IsNaN(s.Latent[i]) || math.IsInf(s.Latent[i], 0) || math.Abs(v-s.Latent[i]) > 1e-9 {
			return base, nil, false, fmt.Errorf("candidate transform differs")
		}
	}
	timing := [2]float64{float64(o.AgeMS) / 300, float64(o.PreviousCommand.Msec) / 100}
	for i, v := range timing {
		if math.IsNaN(r.Timing[i]) || math.IsInf(r.Timing[i], 0) || math.Abs(v-r.Timing[i]) > 1e-9 {
			return base, nil, false, fmt.Errorf("candidate timing differs")
		}
	}
	return base, c, true, nil
}

// Neural evaluation is called only by live Decide/Review. Offline exports use
// jointAimSample and a dedicated CUDA scorer for all learned probabilities.
func (p *PPO) jointAimLogits(o Observation, base Sample) ([]float64, error) {
	f, err := FeaturesForVersion(o, EquippedWeaponFeatureVersion)
	if err != nil {
		return nil, err
	}
	c, err := GeometryActionInputs(f, base)
	if err != nil {
		return nil, err
	}
	if !jointAimActive(c) {
		return nil, nil
	}
	if probe := p.file.DiagnosticAimProbe; probe != nil {
		if probe.Version == DiagnosticPolicyAimProbeVersion {
			if o.Identity.Frame == probe.Frame {
				return append([]float64(nil), probe.Logits...), nil
			}
		} else {
			// Validation proved all final-layer weights are exactly zero. Its
			// logits outside the selected frame are its biases, independent of
			// hidden activations/hints; no neural shortcut applies to normal models.
			logits := p.file.AimCandidateHead.Layers[1].Bias
			if o.Identity.Frame == probe.Frame {
				logits = probe.Logits
			}
			return append([]float64(nil), logits...), nil
		}
	}
	timing := [2]float64{float64(o.AgeMS) / 300, float64(o.PreviousCommand.Msec) / 100}
	hints := make([]float64, 25)
	causals := make([][]float64, 25)
	geometryHead := p.file.AimCandidateHead.Shared != nil && p.file.AimCandidateHead.Shared.Kind == GeometryDeltaAimKind
	geometries := make([][]float64, 25)
	for k := range hints {
		candidate := base
		candidate.Latent, err = ApplyJointAimOffsets(base.Latent, base.AimMode, k)
		if err != nil {
			return nil, err
		}
		causal, e := GeometryActionInputs(f, candidate)
		if e != nil {
			return nil, e
		}
		causals[k] = causal
		if geometryHead {
			hints[k], geometries[k], err = scoreGeometryAimWithGeometry(p.file.AimCandidateHead.Predictor, causal, timing)
		} else {
			hints[k], err = scoreGeometryAim(p.file.AimCandidateHead.Predictor, causal, timing)
		}
		if err != nil {
			return nil, err
		}
	}
	if shared := p.file.AimCandidateHead.Shared; shared != nil {
		logits := make([]float64, 25)
		predictor := p.file.AimCandidateHead.Predictor
		for k, causal := range causals {
			x := make([]float64, 0, 191)
			prob := sigmoid(hints[k])
			baseline := 0.
			if k == 0 {
				baseline = 1
			}
			if geometryHead {
				for j, v := range geometries[k] {
					x = append(x, math.Max(-20, math.Min(20, (v-geometries[0][j])/predictor.Std[185+j])))
				}
				dy, dp := p.file.AimCandidateHead.Offsets[k/5], p.file.AimCandidateHead.Offsets[k%5]
				x = append(x, prob-sigmoid(hints[0]), (dy*dy+dp*dp)/25)
			} else {
				for j, v := range causal {
					x = append(x, math.Max(-20, math.Min(20, (v-predictor.Mean[j])/predictor.Std[j])))
				}
				x = append(x, p.file.AimCandidateHead.Offsets[k/5]/5, p.file.AimCandidateHead.Offsets[k%5]/5, prob, sigmoid(hints[0]), prob-sigmoid(hints[0]), baseline)
			}
			for i, l := range shared.Layers {
				y := make([]float64, len(l.Bias))
				for r, row := range l.Weight {
					y[r] = l.Bias[r]
					for j, w := range row {
						y[r] += w * x[j]
					}
					if i == 0 {
						y[r] = math.Max(0, y[r])
					}
				}
				x = y
			}
			logits[k] = math.Max(-20, math.Min(20, x[0])) + shared.Beta*baseline
			if math.IsNaN(logits[k]) || math.IsInf(logits[k], 0) {
				return nil, fmt.Errorf("nonfinite shared candidate logits")
			}
		}
		return logits, nil
	}
	x, err := JointAimCandidateInputs(f, c, hints, timing)
	if err != nil {
		return nil, err
	}
	for i, l := range p.file.AimCandidateHead.Layers {
		y := make([]float64, len(l.Bias))
		for r, row := range l.Weight {
			y[r] = l.Bias[r]
			for j, w := range row {
				y[r] += w * x[j]
			}
			if i == 0 {
				y[r] = math.Max(0, y[r])
			}
			if math.IsNaN(y[r]) || math.IsInf(y[r], 0) {
				return nil, fmt.Errorf("nonfinite candidate logits")
			}
		}
		x = y
	}
	return x, nil
}

func candidateLogProbabilities(logits []float64) []float64 {
	max := math.Inf(-1)
	for _, v := range logits {
		max = math.Max(max, v)
	}
	sum := 0.
	for _, v := range logits {
		sum += math.Exp(v - max)
	}
	out := make([]float64, len(logits))
	for i, v := range logits {
		out[i] = v - max - math.Log(sum)
	}
	return out
}

func (p *PPO) selectJointAim(o Observation, s *Sample) error {
	if p.file.AimCandidateHead == nil {
		return nil
	}
	logits, err := p.jointAimLogits(o, *s)
	if err != nil || logits == nil {
		return err
	}
	lp := candidateLogProbabilities(logits)
	gate, u := JointAimUniforms(p.continuationSamplingSeed(o), o.Identity.Frame, o.Identity.Life)
	k := 0
	if gate >= math.Exp(lp[0]) {
		tail := candidateLogProbabilities(logits[1:])
		cdf := 0.
		k = 24
		for j, v := range tail {
			cdf += math.Exp(v)
			if u < cdf {
				k = j + 1
				break
			}
		}
	}
	s.AimCandidate = &AimCandidateSample{SourceLatent: s.Latent, Candidate: k, Candidates: 25, LogProbability: lp[k], Timing: [2]float64{float64(o.AgeMS) / 300, float64(o.PreviousCommand.Msec) / 100}}
	if p.file.AimCandidateHead.Shared != nil {
		s.AimCandidate.Logits = append([]float64(nil), logits...)
	}
	s.Latent, err = ApplyJointAimOffsets(s.Latent, s.AimMode, k)
	return err
}

func (p *PPO) jointAimReview(o Observation, s Sample) ([4]float64, float64, error) {
	base, _, active, err := p.jointAimSample(o, s)
	if err != nil {
		return base.Latent, 0, err
	}
	if !active {
		return base.Latent, 0, nil
	}
	logits, err := p.jointAimLogits(o, base)
	if err != nil {
		return base.Latent, 0, err
	}
	lp := candidateLogProbabilities(logits)[s.AimCandidate.Candidate]
	if math.Abs(lp-s.AimCandidate.LogProbability) > 1e-7 {
		return base.Latent, 0, fmt.Errorf("behavior candidate probability differs")
	}
	return base.Latent, lp, nil
}
