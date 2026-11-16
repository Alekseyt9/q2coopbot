package policy

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
)

const HJEPAActorDistribution = "hjepa_primitive_distribution_v1"
const HJEPAActorSampleVersion = "hjepa_primitive_sample_v1"

type HJEPAActorSampling struct {
	Version     string    `json:"version"`
	Seed        int64     `json:"seed"`
	LogStd      []float64 `json:"log_std"`
	Temperature float64   `json:"categorical_temperature"`
}

func validateHJEPAActorSampling(f HJEPAFile) error {
	s := f.PrimitiveSampling
	if s == nil {
		return nil
	}
	if s.Version != HJEPAActorDistribution || s.Seed < 0 || len(s.LogStd) != 4 || f.Planner != "bc" || len(f.Motor) != 0 || f.WeaponOptionFrames != 0 ||
		math.IsNaN(s.Temperature) || math.IsInf(s.Temperature, 0) || s.Temperature < .05 || s.Temperature > 5 {
		return fmt.Errorf("invalid H-JEPA primitive sampling contract")
	}
	for _, v := range s.LogStd {
		if math.IsNaN(v) || math.IsInf(v, 0) || v < -10 || v > 1 {
			return fmt.Errorf("invalid H-JEPA primitive log std")
		}
	}
	return nil
}

func hActorUniform(seed int64, id Identity, stream int) float64 {
	key, _ := json.Marshal(struct {
		Seed     int64
		Identity Identity
		Stream   int
	}{seed, id, stream})
	h := sha256.Sum256(key)
	//52bits plus half a unit keeps both endpoints strictly outside the interval.
	return (float64(binary.LittleEndian.Uint64(h[:8])>>12) + .5) / float64(uint64(1)<<52)
}

func hActorProbabilities(raw []float64, temperature float64) (float64, [3]float64) {
	x := raw[4] / temperature
	fire := 0.
	if x >= 0 {
		fire = 1 / (1 + math.Exp(-x))
	} else {
		e := math.Exp(x)
		fire = e / (1 + e)
	}
	var pose [3]float64
	top := math.Max(raw[5], math.Max(raw[6], raw[7])) / temperature
	total := 0.
	for i := range pose {
		pose[i] = math.Exp(raw[5+i]/temperature - top)
		total += pose[i]
	}
	for i := range pose {
		pose[i] /= total
	}
	return fire, pose
}

func hLogSigmoid(x float64) float64 { return -math.Max(-x, 0) - math.Log1p(math.Exp(-math.Abs(x))) }

// Density is measured in the unsquashed Gaussian latent, as in the existing PPO
// actor. The fixed tanh Jacobian cancels in policy ratios; raw draws are retained.
func hActorLogProbability(raw []float64, cfg HJEPAActorSampling, sample Sample) (float64, error) {
	if len(raw) != 8 || sample.Version != HJEPAActorSampleVersion || sample.SamplingSeed != cfg.Seed || sample.Vertical < 0 || sample.Vertical > 2 {
		return 0, fmt.Errorf("invalid H-JEPA primitive sample")
	}
	lp := 0.
	for i, z := range sample.Latent {
		if math.IsNaN(z) || math.IsInf(z, 0) {
			return 0, fmt.Errorf("nonfinite H-JEPA primitive latent")
		}
		d := (z - raw[i]) / math.Exp(cfg.LogStd[i])
		lp += -.5*d*d - cfg.LogStd[i] - .5*math.Log(2*math.Pi)
	}
	x := raw[4] / cfg.Temperature
	if sample.Attack {
		lp += hLogSigmoid(x)
	} else {
		lp += hLogSigmoid(-x)
	}
	top := math.Max(raw[5], math.Max(raw[6], raw[7])) / cfg.Temperature
	total := 0.
	for i := 5; i < 8; i++ {
		total += math.Exp(raw[i]/cfg.Temperature - top)
	}
	lp += raw[5+sample.Vertical]/cfg.Temperature - top - math.Log(total)
	if math.IsNaN(lp) || math.IsInf(lp, 0) {
		return 0, fmt.Errorf("nonfinite H-JEPA primitive likelihood")
	}
	return lp, nil
}

func (p *HJEPA) samplePrimitive(o Observation, raw []float64) ([]float64, *Sample, error) {
	cfg := p.file.PrimitiveSampling
	sample := &Sample{Version: HJEPAActorSampleVersion, SamplingSeed: cfg.Seed}
	for i := range sample.Latent {
		u, v := hActorUniform(cfg.Seed, o.Identity, 2*i), hActorUniform(cfg.Seed, o.Identity, 2*i+1)
		noise := math.Sqrt(-2*math.Log(u)) * math.Cos(2*math.Pi*v)
		sample.Latent[i] = raw[i] + math.Exp(cfg.LogStd[i])*noise
	}
	fire, pose := hActorProbabilities(raw, cfg.Temperature)
	sample.Attack = hActorUniform(cfg.Seed, o.Identity, 8) < fire
	u := hActorUniform(cfg.Seed, o.Identity, 9)
	for i, prob := range pose {
		sample.Vertical = i
		u -= prob
		if u < 0 {
			break
		}
	}
	lp, err := hActorLogProbability(raw, *cfg, *sample)
	if err != nil {
		return nil, nil, err
	}
	sample.LogProbability = lp
	primitive := make([]float64, 8)
	for i, z := range sample.Latent {
		primitive[i] = math.Tanh(z)
	}
	if sample.Attack {
		primitive[4] = 1
	}
	primitive[5+sample.Vertical] = 1
	return primitive, sample, nil
}

// PrimitiveLogProbability evaluates a recorded native proposal without drawing
// RNG or changing held choices. Seed/format checks bind the behavior receipt.
func (p *HJEPA) PrimitiveLogProbability(o Observation, sample Sample) (float64, error) {
	if p.file.PrimitiveSampling == nil || o.Version != ObservationVersion || o.Health <= 0 || o.AgeMS < 0 || o.AgeMS > 300 {
		return 0, fmt.Errorf("H-JEPA primitive likelihood requires valid actor observation")
	}
	x, err := HJEPAFeatures(o, p.file.Features)
	if err != nil {
		return 0, err
	}
	z, err := hNet(p.file.Levels[0].Encoder, x)
	if err != nil {
		return 0, err
	}
	raw, err := hNet(p.file.Actor, z)
	if err != nil {
		return 0, err
	}
	lp, err := hActorLogProbability(raw, *p.file.PrimitiveSampling, sample)
	if err != nil {
		return 0, err
	}
	if p.file.WeaponHead != "" {
		if sample.Weapon < 0 || sample.Weapon >= len(weaponNames) {
			return 0, fmt.Errorf("invalid H-JEPA sample weapon")
		}
		if p.file.WeaponSamplingSeed != nil {
			probs, err := p.WeaponDistribution(o)
			if err != nil {
				return 0, err
			}
			if probs[sample.Weapon] <= 0 {
				return 0, fmt.Errorf("unavailable sampled weapon")
			}
			lp += math.Log(probs[sample.Weapon])
		} else {
			choice, err := p.chooseWeapon(o, x)
			if err != nil {
				return 0, err
			}
			if choice != sample.Weapon {
				return 0, fmt.Errorf("greedy weapon sample mismatch")
			}
		}
	} else if sample.Weapon != 0 {
		return 0, fmt.Errorf("weapon sample without head")
	}
	return lp, nil
}
