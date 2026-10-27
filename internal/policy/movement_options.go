package policy

import (
	"fmt"
	"math"
	"reflect"
)

const MovementOptionsVersion = "combat_five_frame_movement_options_v1"
const MovementOptionsDistribution = "combat_joint_aim_movement_options_v3"
const MovementOptionsSampler = "frame_hash_movement_options_v1"
const movementOptionsSalt int64 = 0x3c6ef372fe94f82b

var movementOptionShifts = [5][2]float64{{0, 0}, {-1, 0}, {1, 0}, {0, -1}, {0, 1}}

type MovementOptionsFile struct {
	Version  string        `json:"version"`
	Sampler  string        `json:"sampler"`
	Enabled  bool          `json:"enabled"`
	Duration int           `json:"duration_frames"`
	Shifts   [5][2]float64 `json:"mean_shifts"`
	Actor    []DenseLayer  `json:"actor"`
	Value    []DenseLayer  `json:"value"`
}

type MovementOptionSample struct {
	Version        string    `json:"version"`
	Started        int       `json:"started_frame"`
	Decision       bool      `json:"decision"`
	Choice         int       `json:"choice"`
	Uniform        float64   `json:"uniform"`
	Logits         []float64 `json:"logits,omitempty"`
	LogProbability float64   `json:"log_probability"`
	Value          float64   `json:"value"`
}

type movementOptionState struct {
	Identity        Identity
	Started, Choice int
	Valid           bool
}

func validateMovementOptions(f PPOFile) error {
	h := f.MovementOptions
	if h == nil {
		if f.ActionDistribution == MovementOptionsDistribution {
			return fmt.Errorf("movement options require head")
		}
		return nil
	}
	if f.ActionDistribution != MovementOptionsDistribution || h.Version != MovementOptionsVersion || h.Sampler != MovementOptionsSampler ||
		h.Duration != 5 || h.Shifts != movementOptionShifts || f.Features != MemoryRouteFeatureVersion || f.Attention == nil ||
		f.Memory != nil || f.EntityAttention != nil || f.Deterministic || f.AimCandidateHead == nil || f.ConditionalFire == nil ||
		f.WideAim != nil || f.DiagnosticAimProbe != nil || f.DiagnosticMovementProbe != nil || f.DiagnosticContinuation != nil ||
		f.DiagnosticWeaponProbe != nil || f.DiagnosticAnchorProbe != nil || f.DiagnosticWideGate != nil ||
		f.TargetSampling != nil || f.MovementInertia != nil || f.ExecutionMode != "" ||
		f.GeometryAimExplore != nil || f.GeometryAimRanker != nil {
		return fmt.Errorf("invalid movement options contract")
	}
	for _, net := range []struct {
		layers []DenseLayer
		output int
	}{{h.Actor, 5}, {h.Value, 1}} {
		width := 1635 // v13 1554 features + 81 causal actor outputs.
		if len(net.layers) != 3 {
			return fmt.Errorf("movement option network depth")
		}
		for i, n := range []int{32, 16, net.output} {
			layer := net.layers[i]
			if len(layer.Weight) != n || len(layer.Bias) != n {
				return fmt.Errorf("movement option network width")
			}
			for j, row := range layer.Weight {
				if len(row) != width {
					return fmt.Errorf("movement option input width")
				}
				for _, v := range append(append([]float64{}, row...), layer.Bias[j]) {
					if math.IsNaN(v) || math.IsInf(v, 0) || math.Abs(v) > 1e4 {
						return fmt.Errorf("nonfinite movement option weights")
					}
				}
			}
			width = n
		}
	}
	return nil
}

func movementOptionStart(state movementOptionState, id Identity) bool {
	return !state.Valid || !SameLife(state.Identity, id) || id.Frame != state.Identity.Frame+1 || id.Frame == state.Started+5
}

func movementOptionDense(layers []DenseLayer, x []float64) ([]float64, error) {
	for i, l := range layers {
		y := make([]float64, len(l.Bias))
		for j, row := range l.Weight {
			y[j] = l.Bias[j]
			for k, w := range row {
				y[j] += w * x[k]
			}
			if i < 2 {
				y[j] = math.Max(0, y[j])
			}
			if math.IsNaN(y[j]) || math.IsInf(y[j], 0) {
				return nil, fmt.Errorf("nonfinite movement option inference")
			}
		}
		x = y
	}
	return x, nil
}

func (p *PPO) movementOptionLogits(o Observation, raw []float64) ([]float64, float64, error) {
	x, err := FeaturesForVersion(o, p.file.Features)
	if err != nil {
		return nil, 0, err
	}
	if len(x) != 1554 || len(raw) != 81 {
		return nil, 0, fmt.Errorf("movement option context width")
	}
	x = append(x, raw...)
	logits, err := movementOptionDense(p.file.MovementOptions.Actor, x)
	if err != nil {
		return nil, 0, err
	}
	value, err := movementOptionDense(p.file.MovementOptions.Value, x)
	if err != nil {
		return nil, 0, err
	}
	for i, v := range logits {
		logits[i] = math.Max(-20, math.Min(20, v))
	}
	return logits, value[0], nil
}

func movementOptionCategory(logits []float64, uniform float64) (int, []float64) {
	max := math.Inf(-1)
	for _, v := range logits {
		max = math.Max(max, v)
	}
	sum := 0.
	for _, v := range logits {
		sum += math.Exp(v - max)
	}
	lp := make([]float64, len(logits))
	choice := len(logits) - 1
	left := uniform
	for i, v := range logits {
		lp[i] = v - max - math.Log(sum)
	}
	for i, v := range lp {
		left -= math.Exp(v)
		if left <= 0 {
			choice = i
			break
		}
	}
	return choice, lp
}

func (p *PPO) prepareMovementOption(o Observation, raw []float64) (*MovementOptionSample, movementOptionState, error) {
	h := p.file.MovementOptions
	if h == nil || !h.Enabled {
		return nil, movementOptionState{}, nil
	}
	state := p.movementOption
	r := &MovementOptionSample{Version: h.Version, Started: state.Started, Choice: state.Choice}
	if movementOptionStart(state, o.Identity) {
		logits, value, err := p.movementOptionLogits(o, raw)
		if err != nil {
			return nil, state, err
		}
		u, _ := JointAimUniforms(p.file.SamplingSeed^movementOptionsSalt, o.Identity.Frame, o.Identity.Life)
		choice, lp := movementOptionCategory(logits, u)
		r.Started, r.Decision, r.Choice, r.Uniform, r.Logits, r.LogProbability, r.Value = o.Identity.Frame, true, choice, u, logits, lp[choice], value
	}
	return r, movementOptionState{Identity: o.Identity, Started: r.Started, Choice: r.Choice, Valid: true}, nil
}

func (p *PPO) reviewMovementOption(o Observation, raw []float64, r *MovementOptionSample) ([2]float64, float64, error) {
	h := p.file.MovementOptions
	if h == nil || !h.Enabled {
		if r != nil {
			return [2]float64{}, 0, fmt.Errorf("unexpected movement option event")
		}
		return [2]float64{}, 0, nil
	}
	if r == nil || r.Version != h.Version || r.Choice < 0 || r.Choice >= 5 || r.Started < 0 || r.Started > o.Identity.Frame || o.Identity.Frame-r.Started >= 5 || r.Decision != (r.Started == o.Identity.Frame) {
		return [2]float64{}, 0, fmt.Errorf("invalid movement option receipt")
	}
	if !r.Decision {
		if len(r.Logits) != 0 || r.Uniform != 0 || r.LogProbability != 0 || r.Value != 0 {
			return [2]float64{}, 0, fmt.Errorf("held option repeats decision density")
		}
		return h.Shifts[r.Choice], 0, nil
	}
	logits, value, err := p.movementOptionLogits(o, raw)
	if err != nil {
		return [2]float64{}, 0, err
	}
	u, _ := JointAimUniforms(p.file.SamplingSeed^movementOptionsSalt, o.Identity.Frame, o.Identity.Life)
	choice, lp := movementOptionCategory(logits, u)
	if !reflect.DeepEqual(logits, r.Logits) || u != r.Uniform || choice != r.Choice || math.Abs(lp[choice]-r.LogProbability) > 1e-10 || math.Abs(value-r.Value) > 1e-10 {
		return [2]float64{}, 0, fmt.Errorf("movement option decision differs")
	}
	return h.Shifts[choice], lp[choice], nil
}
