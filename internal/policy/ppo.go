package policy

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"math/rand"
	"os"
)

const PPOKind = "combat_ppo_v1"

type PPOFile struct {
	Kind            string               `json:"kind"`
	Features        string               `json:"feature_version"`
	Actor           []DenseLayer         `json:"actor"`
	Value           []DenseLayer         `json:"value"`
	LogStd          [4]float64           `json:"log_std"`
	SamplingSeed    int64                `json:"sampling_seed"`
	Deterministic   bool                 `json:"deterministic"`
	Memory          *GRUFile             `json:"memory,omitempty"`
	Attention       *AttentionFile       `json:"attention,omitempty"`
	EntityAttention *EntityAttentionFile `json:"entity_attention,omitempty"`
	WeaponHead      string               `json:"weapon_head,omitempty"`
	TargetHead      string               `json:"target_head,omitempty"`
}

// Latent-space log probability is used for PPO ratios: the fixed tanh map's
// Jacobian cancels. Quantization/guards belong to environment execution.
type Sample struct {
	Version        string        `json:"version"`
	SamplingSeed   int64         `json:"sampling_seed"`
	Latent         [4]float64    `json:"latent"`
	Attack         bool          `json:"attack"`
	Vertical       int           `json:"vertical"`
	Weapon         int           `json:"weapon,omitempty"`
	Target         int           `json:"target,omitempty"`
	LogProbability float64       `json:"log_probability"`
	Value          float64       `json:"value"`
	Memory         *MemorySample `json:"memory,omitempty"`
}
type PPO struct {
	file          PPOFile
	actor, critic *MLP
	version       string
	rng           *rand.Rand
	last          *Sample
	memory        memoryState
	lastAction    *Action
}

func LoadPPO(path string) (*PPO, error) {
	b, e := os.ReadFile(path)
	if e != nil {
		return nil, e
	}
	if len(b) > MaxModelBytes {
		return nil, fmt.Errorf("PPO file too large")
	}
	d := json.NewDecoder(bytes.NewReader(b))
	d.DisallowUnknownFields()
	var f PPOFile
	if e = d.Decode(&f); e != nil {
		return nil, e
	}
	if d.Decode(new(any)) != io.EOF {
		return nil, fmt.Errorf("trailing PPO data")
	}
	if f.Kind != PPOKind || (f.Features != FeatureVersion && f.Features != AimFeatureVersion && f.Features != BBoxFeatureVersion && f.Features != TypedFeatureVersion && f.Features != RecoilFeatureVersion && f.Features != WeaponFeatureVersion && f.Features != TargetFeatureVersion) || f.SamplingSeed < 0 {
		return nil, fmt.Errorf("invalid PPO header")
	}
	outputs := 8
	if f.WeaponHead != "" {
		if f.WeaponHead != WeaponHeadVersion || (f.Features != WeaponFeatureVersion && f.Features != TargetFeatureVersion) || f.EntityAttention != nil {
			return nil, fmt.Errorf("invalid PPO weapon contract")
		}
		outputs += len(weaponNames)
	}
	if f.TargetHead != "" {
		if f.TargetHead != TargetHeadVersion || f.Features != TargetFeatureVersion || f.WeaponHead != WeaponHeadVersion || f.EntityAttention != nil {
			return nil, fmt.Errorf("invalid PPO target contract")
		}
		outputs = TargetOutputWidth
	} else if f.Features == TargetFeatureVersion {
		return nil, fmt.Errorf("target features require target head")
	}
	if e = validateFeatureLayers(f.Actor, outputs, f.Features); e != nil {
		return nil, e
	}
	if e = validateFeatureLayers(f.Value, 1, f.Features); e != nil {
		return nil, e
	}
	if e = validateMemory(f.Memory, f.Actor, f.Value); e != nil {
		return nil, e
	}
	if e = validateAttention(f.Attention, f.Memory, f.Actor, f.Value); e != nil {
		return nil, e
	}
	if e = validateEntityAttention(f.EntityAttention, f); e != nil {
		return nil, e
	}
	for _, s := range f.LogStd {
		if math.IsNaN(s) || math.IsInf(s, 0) || s < -8 || s > 1 {
			return nil, fmt.Errorf("invalid PPO standard deviation")
		}
	}
	canonical := f
	canonical.SamplingSeed = 0
	canonical.Deterministic = false
	data, _ := json.Marshal(canonical)
	h := sha256.Sum256(data)
	p := &PPO{file: f, version: "ppo:" + hex.EncodeToString(h[:]), rng: rand.New(rand.NewSource(f.SamplingSeed))}
	p.actor = &MLP{file: MLPFile{Features: f.Features, Layers: f.Actor}}
	p.critic = &MLP{file: MLPFile{Features: f.Features, Layers: f.Value}}
	return p, nil
}
func (p *PPO) Version() string           { return p.version }
func (p *PPO) FeatureVersion() string    { return p.file.Features }
func (p *PPO) WeaponHeadVersion() string { return p.file.WeaponHead }
func (p *PPO) SamplingSeed() int64       { return p.file.SamplingSeed }
func (p *PPO) IsStochastic() bool        { return !p.file.Deterministic }
func (p *PPO) LastSample() *Sample       { return p.last }
func (p *PPO) IsRecurrent() bool         { return p.file.Memory != nil || p.file.Attention != nil }
func (p *PPO) MemoryVersion() string {
	if p.file.Attention != nil {
		return AttentionVersion
	}
	if p.file.Memory != nil {
		return GRUVersion
	}
	return ""
}
func (p *PPO) Value(o Observation) (float64, error) {
	if p.file.EntityAttention != nil {
		_, v, _, e := p.memoryRaw(o, nil)
		return v, e
	}
	if p.IsRecurrent() {
		_, v, _, e := p.memoryRaw(o, p.beforeMemory(o))
		return v, e
	}
	x, e := p.critic.Raw(o)
	if e != nil {
		return 0, e
	}
	return x[0], nil
}
func softplus(x float64) float64 { return math.Max(x, 0) + math.Log1p(math.Exp(-math.Abs(x))) }
func (p *PPO) Review(o Observation, s Sample) (Action, float64, float64, error) {
	if s.Version != p.version || s.Vertical < 0 || s.Vertical > 2 {
		return Action{}, 0, 0, fmt.Errorf("PPO sample version/category")
	}
	if p.file.WeaponHead == "" && s.Weapon != 0 {
		return Action{}, 0, 0, fmt.Errorf("weapon sample on legacy policy")
	}
	if p.file.TargetHead == "" && s.Target != 0 {
		return Action{}, 0, 0, fmt.Errorf("target sample on legacy policy")
	}
	x, value, _, e := p.memoryRaw(o, s.Memory)
	if e != nil {
		return Action{}, 0, 0, e
	}
	lp := 0.0
	means := [4]float64{x[0], x[1], x[2], x[3]}
	if p.file.TargetHead != "" {
		probs, err := targetLogProbabilities(o, x[TargetLogitOffset:TargetAimOffset])
		if err != nil {
			return Action{}, 0, 0, err
		}
		if s.Target < 0 || s.Target >= len(probs) || math.IsInf(probs[s.Target], -1) {
			return Action{}, 0, 0, fmt.Errorf("unavailable target sample")
		}
		means, err = targetMeans(x, s.Target)
		if err != nil {
			return Action{}, 0, 0, err
		}
		lp += probs[s.Target]
	}
	for i, z := range s.Latent {
		if math.IsNaN(z) || math.IsInf(z, 0) {
			return Action{}, 0, 0, fmt.Errorf("nonfinite latent")
		}
		delta := (z - means[i]) / math.Exp(p.file.LogStd[i])
		lp += -.5*delta*delta - p.file.LogStd[i] - .5*math.Log(2*math.Pi)
	}
	if s.Attack {
		lp -= softplus(-x[4])
	} else {
		lp -= softplus(x[4])
	}
	max := math.Max(x[5], math.Max(x[6], x[7]))
	sum := 0.0
	for _, v := range x[5:8] {
		sum += math.Exp(v - max)
	}
	lp += x[5+s.Vertical] - max - math.Log(sum)
	a := Action{Version: ActionVersion, Identity: o.Identity, Forward: math.Tanh(s.Latent[0]), Side: math.Tanh(s.Latent[1]), YawDelta: 180 * math.Tanh(s.Latent[2]), PitchDelta: 180 * math.Tanh(s.Latent[3]), Attack: s.Attack, Vertical: []string{"release", "jump", "crouch"}[s.Vertical]}
	if p.file.TargetHead != "" && s.Target > 0 {
		enemy := TargetEnemies(o)[s.Target-1]
		a.TargetEntity = enemy.ID
		if enemy.Track != nil {
			a.TargetTrack = *enemy.Track
		}
	}
	if p.file.WeaponHead != "" {
		probs, err := weaponLogProbabilities(o, x[8:20])
		if err != nil {
			return Action{}, 0, 0, err
		}
		if s.Weapon < 0 || s.Weapon >= len(probs) || math.IsInf(probs[s.Weapon], -1) {
			return Action{}, 0, 0, fmt.Errorf("unavailable weapon sample")
		}
		lp += probs[s.Weapon]
		a.Weapon = weaponNames[s.Weapon]
	}
	_, e = Command(o, a, [3]int16{})
	return a, lp, value, e
}
func (p *PPO) Decide(o Observation) (Action, error) {
	if p.IsRecurrent() && p.memory.valid && p.memory.identity == o.Identity && p.lastAction != nil {
		return *p.lastAction, nil
	}
	p.last = nil
	before := p.beforeMemory(o)
	x, _, after, e := p.memoryRaw(o, before)
	if e != nil {
		return Action{}, e
	}
	s := Sample{Version: p.version, SamplingSeed: p.file.SamplingSeed, Memory: before}
	means := [4]float64{x[0], x[1], x[2], x[3]}
	if p.file.TargetHead != "" {
		probs, err := targetLogProbabilities(o, x[TargetLogitOffset:TargetAimOffset])
		if err != nil {
			return Action{}, err
		}
		if p.file.Deterministic {
			for i, lp := range probs {
				if lp > probs[s.Target] {
					s.Target = i
				}
			}
		} else {
			u := p.rng.Float64()
			for i, lp := range probs {
				if math.IsInf(lp, -1) {
					continue
				}
				s.Target = i // Last available category covers rounding at the CDF tail.
				u -= math.Exp(lp)
				if u <= 0 {
					break
				}
			}
		}
		means, err = targetMeans(x, s.Target)
		if err != nil {
			return Action{}, err
		}
	}
	for i := range s.Latent {
		s.Latent[i] = means[i]
		if !p.file.Deterministic {
			s.Latent[i] += math.Exp(p.file.LogStd[i]) * p.rng.NormFloat64()
		}
	}
	s.Attack = x[4] >= 0
	s.Vertical = 0
	for i := 1; i < 3; i++ {
		if x[5+i] > x[5+s.Vertical] {
			s.Vertical = i
		}
	}
	if !p.file.Deterministic {
		s.Attack = p.rng.Float64() < math.Exp(-softplus(-x[4]))
		max := math.Max(x[5], math.Max(x[6], x[7]))
		sum := 0.0
		for _, v := range x[5:8] {
			sum += math.Exp(v - max)
		}
		u := p.rng.Float64() * sum
		for i, v := range x[5:8] {
			u -= math.Exp(v - max)
			if u <= 0 {
				s.Vertical = i
				break
			}
		}
	}
	if p.file.WeaponHead != "" {
		probs, err := weaponProbabilities(o, x[8:20])
		if err != nil {
			return Action{}, err
		}
		if p.file.Deterministic {
			for i, probability := range probs {
				if probability > probs[s.Weapon] {
					s.Weapon = i
				}
			}
		} else {
			u := p.rng.Float64()
			for i, probability := range probs {
				if probability == 0 {
					continue
				}
				s.Weapon = i
				u -= probability
				if u <= 0 {
					break
				}
			}
		}
	}
	a, lp, value, e := p.Review(o, s)
	if e != nil {
		return Action{}, e
	}
	s.LogProbability, s.Value = lp, value
	if p.IsRecurrent() {
		p.memory = memoryState{identity: o.Identity, actor: after.Actor, value: after.Value, position: after.Position, valid: true}
		p.lastAction = &a
	}
	if !p.file.Deterministic {
		p.last = &s
	}
	return a, nil
}
