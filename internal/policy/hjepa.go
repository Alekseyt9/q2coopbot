package policy

// HJEPA is an experimental structured-observation adaptation, not the authors'
// visual checkpoint. Each level has its own latent space and temporal stride.
import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
)

const HJEPAKind = "combat_hjepa_v1"

type HJEPALevel struct {
	Stride    int          `json:"stride"`
	Encoder   []DenseLayer `json:"encoder"`
	Predictor []DenseLayer `json:"predictor"`
	Utility   []DenseLayer `json:"utility"`
}
type HJEPAFile struct {
	PrimitiveSampling  *HJEPAActorSampling `json:"primitive_sampling,omitempty"`
	WorldActionSpace   string              `json:"world_action_space,omitempty"` // explicit predictor inputs; empty preserves historical files
	WeaponHead         string              `json:"weapon_head,omitempty"`
	WeaponFeatures     string              `json:"weapon_feature_version,omitempty"`
	WeaponActor        []DenseLayer        `json:"weapon_actor,omitempty"`
	WeaponSamplingSeed *int64              `json:"weapon_sampling_seed,omitempty"` // nil: greedy; present: reproducible categorical
	WeaponOptionFrames int                 `json:"weapon_option_frames,omitempty"` // optional learned choice commitment, 1..64 frames
	WeaponMode         string              `json:"weapon_mode,omitempty"`          // explicit inventory policy declaration
	Planner            string              `json:"planner,omitempty"`              // empty/hierarchy, flat, bc, sequence
	Kind               string              `json:"kind"`
	Features           string              `json:"feature_version"`
	Levels             []HJEPALevel        `json:"levels"`
	Actor              []DenseLayer        `json:"actor"`
	Motor              []DenseLayer        `json:"motor,omitempty"`     // optional independently distilled primitive proposal
	PlanAxes           []int               `json:"plan_axes,omitempty"` // optional search ablation: 0 forward,1 side,2 yaw,3 pitch,4 fire
	PriorWeight        float64             `json:"prior_weight"`
	PlanStep           float64             `json:"plan_step,omitempty"`
	ConsistencyMargin  *float64            `json:"consistency_margin,omitempty"`
}
type HJEPA struct {
	samplingRestore *hSamplingRestore
	lastSample      *Sample
	file            HJEPAFile
	version         string
	lastSearch      HJEPASearchAudit
	weaponOption    hWeaponOption
}

type hWeaponOption struct {
	valid       bool
	last        Identity
	start, slot int
}

// ResetEpisode separates independently loaded native captures during replay.
func (p *HJEPA) ResetEpisode() {
	p.samplingRestore = nil
	p.weaponOption = hWeaponOption{}
	p.lastSearch = HJEPASearchAudit{}
	p.lastSample = nil
}

func (p *HJEPA) LastSample() *Sample { return p.lastSample }

// KeepsEquippedWeapon admits inventory fixtures only by an explicit model declaration.
// Native empty-ammo switching remains engine behavior, not a learned decision.
func (h *HJEPA) KeepsEquippedWeapon() bool { return h.file.WeaponMode == "keep_equipped_v1" }

func (h *HJEPA) WeaponHeadVersion() string { return h.file.WeaponHead }
func (h *HJEPA) SupportsInventory() bool {
	return h.KeepsEquippedWeapon() || h.file.WeaponHead == WeaponHeadVersion
}

// WeaponLogits is also used by offline serialization parity checks.
func (h *HJEPA) WeaponLogits(features []float64) ([]float64, error) {
	if h.file.WeaponHead != WeaponHeadVersion || len(features) != h.weaponInputWidth() {
		return nil, fmt.Errorf("H-JEPA weapon input/header")
	}
	for _, v := range features {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return nil, fmt.Errorf("nonfinite weapon features")
		}
	}
	return hNet(h.file.WeaponActor, features)
}

// WeaponDistribution exposes the behavior likelihood for fresh offline training.
// Held options and greedy choices are not categorical policy-gradient samples.
func (h *HJEPA) WeaponDistribution(o Observation) ([]float64, error) {
	if h.file.WeaponSamplingSeed == nil || h.file.WeaponOptionFrames != 0 {
		return nil, fmt.Errorf("weapon likelihood requires stochastic frame-wise choices")
	}
	x, err := FeaturesForVersion(o, WeaponFeatureVersion)
	if err != nil {
		return nil, err
	}
	if h.file.WeaponFeatures == HJEPAWeaponFeatureVersion {
		x, err = appendHJEPAWeaponContext(o, x)
		if err != nil {
			return nil, err
		}
	}
	logits, err := h.WeaponLogits(x)
	if err != nil {
		return nil, err
	}
	return weaponProbabilities(o, logits)
}

func validateHNet(net []DenseLayer, input, output int) error {
	if len(net) < 1 || len(net) > 3 {
		return fmt.Errorf("H-JEPA network depth")
	}
	for _, l := range net {
		if len(l.Bias) < 1 || len(l.Bias) > MaxHiddenWidth || len(l.Weight) != len(l.Bias) {
			return fmt.Errorf("H-JEPA network width")
		}
		for r, row := range l.Weight {
			if len(row) != input {
				return fmt.Errorf("H-JEPA network input")
			}
			for _, v := range append(append([]float64{}, row...), l.Bias[r]) {
				if math.IsNaN(v) || math.IsInf(v, 0) || math.Abs(v) > 1e4 {
					return fmt.Errorf("H-JEPA nonfinite/excessive weight")
				}
			}
		}
		input = len(l.Bias)
	}
	if input != output {
		return fmt.Errorf("H-JEPA network output")
	}
	return nil
}
func hNet(net []DenseLayer, x []float64) ([]float64, error) {
	for i, l := range net {
		y := make([]float64, len(l.Bias))
		for r, row := range l.Weight {
			y[r] = l.Bias[r]
			for c, w := range row {
				y[r] += w * x[c]
			}
			if i < len(net)-1 {
				y[r] = math.Max(0, y[r])
			}
			if math.IsNaN(y[r]) || math.IsInf(y[r], 0) {
				return nil, fmt.Errorf("nonfinite H-JEPA inference")
			}
		}
		x = y
	}
	return x, nil
}
func LoadHJEPA(path string) (*HJEPA, error) {
	b, e := os.ReadFile(path)
	if e != nil {
		return nil, e
	}
	if len(b) > MaxModelBytes {
		return nil, fmt.Errorf("H-JEPA file too large")
	}
	d := json.NewDecoder(bytes.NewReader(b))
	d.DisallowUnknownFields()
	var f HJEPAFile
	if e = d.Decode(&f); e != nil {
		return nil, e
	}
	if d.Decode(new(any)) != io.EOF {
		return nil, fmt.Errorf("trailing H-JEPA data")
	}
	featureWidth, featureError := HJEPAFeatureWidth(f.Features)
	if f.Kind != HJEPAKind || featureError != nil || len(f.Levels) != 3 || math.IsNaN(f.PriorWeight) || math.IsInf(f.PriorWeight, 0) || f.PriorWeight < .01 || f.PriorWeight > 10 {
		return nil, fmt.Errorf("unsupported H-JEPA header")
	}
	if f.WorldActionSpace != "" && f.WorldActionSpace != "proposed_primitive_v1" {
		return nil, fmt.Errorf("unsupported H-JEPA world action space")
	}
	if f.Planner != "" && f.Planner != "hierarchy" && f.Planner != "flat" && f.Planner != "bc" && f.Planner != "sequence" {
		return nil, fmt.Errorf("unsupported H-JEPA planner")
	}
	if err := validateHJEPAActorSampling(f); err != nil {
		return nil, err
	}
	if f.WeaponMode != "" && f.WeaponMode != "keep_equipped_v1" && f.WeaponMode != "learned_masked_v1" {
		return nil, fmt.Errorf("unsupported H-JEPA weapon mode")
	}
	if f.WeaponMode == "learned_masked_v1" {
		if f.WeaponFeatures != "" && f.WeaponFeatures != HJEPAWeaponFeatureVersion {
			return nil, fmt.Errorf("unsupported H-JEPA weapon features")
		}
		if f.WeaponOptionFrames < 0 || f.WeaponOptionFrames > 64 {
			return nil, fmt.Errorf("invalid weapon option duration")
		}
		if f.WeaponHead != WeaponHeadVersion {
			return nil, fmt.Errorf("H-JEPA weapon head version required")
		}
		weaponWidth := WeaponFeatureWidth
		if f.WeaponFeatures == HJEPAWeaponFeatureVersion {
			weaponWidth = HJEPAWeaponFeatureWidth
		}
		if err := validateHNet(f.WeaponActor, weaponWidth, len(weaponNames)); err != nil {
			return nil, err
		}
		if f.WeaponSamplingSeed != nil && *f.WeaponSamplingSeed < 0 {
			return nil, fmt.Errorf("negative weapon sampling seed")
		}
	} else if f.WeaponHead != "" || f.WeaponFeatures != "" || len(f.WeaponActor) != 0 || f.WeaponSamplingSeed != nil || f.WeaponOptionFrames != 0 {
		return nil, fmt.Errorf("H-JEPA weapon head requires learned mode")
	}
	if math.IsNaN(f.PlanStep) || math.IsInf(f.PlanStep, 0) || f.PlanStep < 0 || f.PlanStep > .5 {
		return nil, fmt.Errorf("invalid H-JEPA plan step")
	}
	if f.ConsistencyMargin != nil {
		m := *f.ConsistencyMargin
		if f.Planner != "sequence" || math.IsNaN(m) || math.IsInf(m, 0) || m < 0 || m > 5 {
			return nil, fmt.Errorf("invalid H-JEPA consistency margin")
		}
	}
	seenAxes := map[int]bool{}
	for _, axis := range f.PlanAxes {
		if axis < 0 || axis > 4 || seenAxes[axis] {
			return nil, fmt.Errorf("invalid H-JEPA search axes")
		}
		seenAxes[axis] = true
	}
	input := featureWidth
	for i, l := range f.Levels {
		if l.Stride != []int{1, 4, 16}[i] || len(l.Encoder) == 0 {
			return nil, fmt.Errorf("H-JEPA strides must be 1,4,16")
		}
		z := len(l.Encoder[len(l.Encoder)-1].Bias)
		if z < 2 || z > 64 {
			return nil, fmt.Errorf("H-JEPA latent width")
		}
		if e = validateHNet(l.Encoder, input, z); e != nil {
			return nil, e
		}
		if e = validateHNet(l.Predictor, z+8*l.Stride, z); e != nil {
			return nil, e
		}
		if e = validateHNet(l.Utility, z, 1); e != nil {
			return nil, e
		}
		input = z
	}
	if e = validateHNet(f.Actor, len(f.Levels[0].Encoder[len(f.Levels[0].Encoder)-1].Bias), 8); e != nil {
		return nil, e
	}
	if len(f.Motor) > 0 {
		if e = validateHNet(f.Motor, featureWidth, 8); e != nil {
			return nil, e
		}
	}
	h := sha256.Sum256(b)
	return &HJEPA{file: f, version: "hjepa:" + hex.EncodeToString(h[:])}, nil
}
func (p *HJEPA) Version() string { return p.version }

// Replay exposes the same native networks to the cross-language audit tool.
func (p *HJEPA) Replay(features, action []float64) (latents [][]float64, actor []float64, predictions [][]float64, err error) {
	width, _ := HJEPAFeatureWidth(p.file.Features)
	if len(features) != width || len(action) != 8 {
		err = fmt.Errorf("H-JEPA replay width")
		return
	}
	x := features
	for i, l := range p.file.Levels {
		x, err = hNet(l.Encoder, x)
		if err != nil {
			return
		}
		latents = append(latents, x)
		in := append([]float64{}, x...)
		for t := 0; t < l.Stride; t++ {
			in = append(in, action...)
		}
		var pred []float64
		pred, err = hNet(p.file.Levels[i].Predictor, in)
		if err != nil {
			return
		}
		predictions = append(predictions, pred)
	}
	actor, err = hNet(p.file.Actor, latents[0])
	if err == nil && len(p.file.Motor) > 0 {
		actor, err = hNet(p.file.Motor, features)
	}
	return
}

// Primitive vectors encode four normalized axes, attack, and vertical one-hot.
func hPrimitive(raw []float64) []float64 {
	a := make([]float64, 8)
	for i := 0; i < 4; i++ {
		a[i] = math.Tanh(raw[i])
	}
	if raw[4] >= 0 {
		a[4] = 1
	}
	best := 5
	for i := 6; i < 8; i++ {
		if raw[i] > raw[best] {
			best = i
		}
	}
	a[best] = 1
	return a
}
func hCandidates(prior []float64) [][]float64 {
	out := [][]float64{append([]float64{}, prior...)}
	for _, q := range [][2]float64{{0, -.5}, {0, .5}, {1, -.5}, {1, .5}, {2, -.025}, {2, .025}, {3, -.025}, {3, .025}} {
		a := append([]float64{}, prior...)
		i := int(q[0])
		a[i] = math.Max(-1, math.Min(1, a[i]+q[1]))
		out = append(out, a)
	}
	a := append([]float64{}, prior...)
	a[4] = 1 - a[4]
	return append(out, a)
}
func (p *HJEPA) candidates(prior []float64) [][]float64 {
	all := hCandidates(prior)
	if p.file.PlanStep > 0 {
		for _, a := range all {
			for i := 0; i < 4; i++ {
				delta := a[i] - prior[i]
				if delta != 0 {
					scale := 1.
					if i >= 2 {
						scale = .05
					}
					a[i] = math.Max(-1, math.Min(1, prior[i]+math.Copysign(p.file.PlanStep*scale, delta)))
				}
			}
		}
	}
	if len(p.file.PlanAxes) == 0 {
		return all
	}
	allowed := map[int]bool{}
	for _, axis := range p.file.PlanAxes {
		allowed[axis] = true
	}
	var out [][]float64
	for _, a := range all {
		valid := true
		for i, v := range a {
			if v != prior[i] && !allowed[i] {
				valid = false
				break
			}
		}
		if valid {
			out = append(out, a)
		}
	}
	return out
}
func (p *HJEPA) Decide(o Observation) (Action, error) {
	p.lastSample = nil
	p.lastSearch = HJEPASearchAudit{}
	if o.Version != ObservationVersion || o.Health <= 0 || o.AgeMS < 0 || o.AgeMS > 300 {
		return Action{}, fmt.Errorf("invalid H-JEPA observation")
	}
	x, e := HJEPAFeatures(o, p.file.Features)
	if e != nil {
		return Action{}, e
	}
	for _, v := range x {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return Action{}, fmt.Errorf("nonfinite H-JEPA observation")
		}
	}
	features := x
	zs := make([][]float64, 3)
	for i, l := range p.file.Levels {
		x, e = hNet(l.Encoder, x)
		if e != nil {
			return Action{}, e
		}
		zs[i] = x
	}
	raw, e := hNet(p.file.Actor, zs[0])
	if e == nil && len(p.file.Motor) > 0 {
		raw, e = hNet(p.file.Motor, features)
	}
	if e != nil {
		return Action{}, e
	}
	prior := hPrimitive(raw)
	if p.file.PrimitiveSampling != nil {
		prior, p.lastSample, e = p.samplePrimitive(o, raw)
		if e != nil {
			return Action{}, e
		}
	}
	candidates := p.candidates(prior)
	var goal, chosen []float64
	chosen = prior
	if p.file.Planner == "sequence" {
		chosen, e = p.sequenceAction(zs, prior)
		if e != nil {
			return Action{}, e
		}
	}
	top := 2
	if p.file.Planner == "flat" {
		top = 0
	}
	if p.file.Planner == "bc" {
		top = -1
	}
	if p.file.Planner == "sequence" {
		top = -1
	}
	for level := top; level >= 0; level-- {
		l := p.file.Levels[level]
		best := math.Inf(1)
		var nextGoal []float64
		for _, a := range candidates {
			in := append([]float64{}, zs[level]...)
			for t := 0; t < l.Stride; t++ {
				in = append(in, a...)
			}
			pred, err := hNet(l.Predictor, in)
			if err != nil {
				return Action{}, err
			}
			value, err := hNet(l.Utility, pred)
			if err != nil {
				return Action{}, err
			}
			score := -value[0]
			if goal != nil {
				projected, err := hNet(p.file.Levels[level+1].Encoder, pred)
				if err != nil {
					return Action{}, err
				}
				score = 0
				for j, v := range projected {
					score += (v - goal[j]) * (v - goal[j]) / float64(len(goal))
				}
			}
			for j, v := range a {
				score += p.file.PriorWeight * (v - prior[j]) * (v - prior[j])
			}
			if score < best {
				best = score
				chosen = a
				nextGoal = pred
			}
		}
		goal = nextGoal
	}
	vertical := 0
	for i := 1; i < 3; i++ {
		if chosen[5+i] > chosen[5+vertical] {
			vertical = i
		}
	}
	a := Action{Version: ActionVersion, Identity: o.Identity, Forward: chosen[0], Side: chosen[1], YawDelta: chosen[2] * 180, PitchDelta: chosen[3] * 180, Attack: chosen[4] >= .5, Vertical: []string{"release", "jump", "crouch"}[vertical]}
	if p.file.WeaponHead != "" {
		best, err := p.chooseWeapon(o, features)
		if err != nil {
			return Action{}, err
		}
		a.Weapon = weaponNames[best]
		if p.lastSample != nil {
			p.lastSample.Weapon = best
			if p.file.WeaponSamplingSeed != nil {
				probs, err := p.WeaponDistribution(o)
				if err != nil {
					return Action{}, err
				}
				p.lastSample.LogProbability += math.Log(probs[best])
			}
		}
	}
	_, e = Command(o, a, [3]int16{})
	return a, e
}

func (p *HJEPA) chooseWeapon(o Observation, features []float64) (int, error) {
	// The weapon head has its own input contract. World context must neither
	// append a second context tail nor change historical 845-wide heads.
	features = features[:WeaponFeatureWidth]
	if p.file.WeaponFeatures == HJEPAWeaponFeatureVersion {
		var err error
		features, err = appendHJEPAWeaponContext(o, features)
		if err != nil {
			return 0, err
		}
	}
	if p.file.WeaponOptionFrames > 0 {
		mask, err := WeaponAvailability(o)
		if err != nil {
			return 0, err
		}
		s := &p.weaponOption
		if s.valid && SameLife(s.last, o.Identity) && o.Identity.Frame >= s.last.Frame && o.Identity.Frame-s.last.Frame <= 1 && o.Identity.Frame-s.start < p.file.WeaponOptionFrames && mask[s.slot] {
			s.last = o.Identity
			return s.slot, nil
		}
	}
	logits, err := p.WeaponLogits(features)
	if err != nil {
		return 0, err
	}
	probs, err := weaponProbabilities(o, logits)
	if err != nil {
		return 0, err
	}
	best := 0
	for i := 1; i < len(probs); i++ {
		if probs[i] > probs[best] {
			best = i
		}
	}
	if p.file.WeaponSamplingSeed != nil {
		// Counter-based sampling: replaying the same observed identity returns
		// the same choice without advancing hidden RNG or consulting a teacher.
		samplingID, err := p.samplingIdentity(o.Identity)
		if err != nil {
			return 0, err
		}
		key, _ := json.Marshal(struct {
			Seed     int64
			Identity Identity
		}{*p.file.WeaponSamplingSeed, samplingID})
		h := sha256.Sum256(key)
		u := float64(binary.LittleEndian.Uint64(h[:8])>>11) / float64(uint64(1)<<53)
		for i, v := range probs {
			if v > 0 {
				best = i
				u -= v
				if u < 0 {
					break
				}
			}
		}
	}
	if p.file.WeaponOptionFrames > 0 {
		p.weaponOption = hWeaponOption{valid: true, last: o.Identity, start: o.Identity.Frame, slot: best}
	}
	return best, nil
}
