package policy

import (
	"fmt"
	"math"
	"reflect"
)

const GRUVersion = "combat_residual_gru_v1"

// Gate ordering and candidate equation match torch.nn.GRU (r,z,n).
type GRUCell struct {
	Input  DenseLayer `json:"input"`
	Hidden DenseLayer `json:"hidden"`
	Output DenseLayer `json:"output"`
}
type GRUFile struct {
	Version string  `json:"version"`
	Actor   GRUCell `json:"actor"`
	Value   GRUCell `json:"value"`
}
type MemorySample struct {
	Actor    []float64 `json:"actor"`
	Value    []float64 `json:"value"`
	Reset    bool      `json:"reset"`
	Position int       `json:"position,omitempty"`
}
type memoryState struct {
	identity     Identity
	actor, value []float64
	valid        bool
	position     int
}

func validMatrix(l DenseLayer, rows, cols int) bool {
	if len(l.Bias) != rows || len(l.Weight) != rows {
		return false
	}
	for i, row := range l.Weight {
		if len(row) != cols {
			return false
		}
		for _, v := range append(append([]float64{}, row...), l.Bias[i]) {
			if math.IsNaN(v) || math.IsInf(v, 0) || math.Abs(v) > 1e4 {
				return false
			}
		}
	}
	return true
}
func validateMemory(f *GRUFile, actor, value []DenseLayer) error {
	if f == nil {
		return nil
	}
	if f.Version != GRUVersion {
		return fmt.Errorf("unsupported recurrent architecture")
	}
	for i, c := range []GRUCell{f.Actor, f.Value} {
		base := actor
		outputs := len(actor[len(actor)-1].Bias)
		if i == 1 {
			base = value
			outputs = 1
		}
		h := len(c.Hidden.Weight) / 3
		if h < 1 || h > 128 || !validMatrix(c.Input, 3*h, len(base[1].Bias)) || !validMatrix(c.Hidden, 3*h, h) || !validMatrix(c.Output, outputs, h) {
			return fmt.Errorf("invalid GRU shape/weights")
		}
	}
	return nil
}
func affine(l DenseLayer, x []float64) []float64 {
	y := append([]float64{}, l.Bias...)
	for i, row := range l.Weight {
		for j, w := range row {
			y[i] += w * x[j]
		}
	}
	return y
}
func sigmoid(x float64) float64 { return math.Exp(-softplus(-x)) }
func gruForward(c GRUCell, x, h []float64) ([]float64, []float64) {
	a, b := affine(c.Input, x), affine(c.Hidden, h)
	n := len(h)
	next := make([]float64, n)
	for i := range h {
		r, z := sigmoid(a[i]+b[i]), sigmoid(a[n+i]+b[n+i])
		candidate := math.Tanh(a[2*n+i] + r*b[2*n+i])
		next[i] = (1-z)*candidate + z*h[i]
	}
	return affine(c.Output, next), next
}
func (p *PPO) beforeMemory(o Observation) *MemorySample {
	if !p.IsRecurrent() {
		return nil
	}
	reset := !p.memory.valid || !SameLife(p.memory.identity, o.Identity) || o.Identity.Frame != p.memory.identity.Frame+1
	a, v := p.memory.actor, p.memory.value
	if p.file.Attention != nil {
		position := p.memory.position
		if reset {
			a = []float64{}
			v = []float64{}
			position = 0
		}
		return &MemorySample{Actor: append([]float64{}, a...), Value: append([]float64{}, v...), Reset: reset, Position: position}
	}
	if reset {
		a = make([]float64, len(p.file.Memory.Actor.Hidden.Weight)/3)
		v = make([]float64, len(p.file.Memory.Value.Hidden.Weight)/3)
	}
	return &MemorySample{Actor: append([]float64{}, a...), Value: append([]float64{}, v...), Reset: reset}
}
func validState(x []float64, n int) bool {
	if len(x) != n {
		return false
	}
	for _, v := range x {
		if math.IsNaN(v) || math.IsInf(v, 0) || math.Abs(v) > 1.000001 {
			return false
		}
	}
	return true
}
func recurrentRaw(base []DenseLayer, c GRUCell, x, h []float64) ([]float64, []float64) {
	for _, l := range base[:2] {
		x = affine(l, x)
		for i := range x {
			x[i] = math.Max(0, x[i])
		}
	}
	y := affine(base[2], x)
	delta, next := gruForward(c, x, h)
	for i := range y {
		y[i] += delta[i]
	}
	return y, next
}
func (p *PPO) memoryRaw(o Observation, m *MemorySample) ([]float64, float64, *MemorySample, error) {
	if !p.IsRecurrent() {
		if m != nil {
			return nil, 0, nil, fmt.Errorf("memory sample for stateless policy")
		}
		if p.file.EntityAttention != nil {
			return p.entityRaw(o)
		}
		a, e := p.actor.Raw(o)
		if e != nil {
			return nil, 0, nil, e
		}
		v, e := p.critic.Raw(o)
		if e != nil {
			return nil, 0, nil, e
		}
		return a, v[0], nil, nil
	}
	if p.file.Attention != nil {
		return p.attentionRaw(o, m)
	}
	f := p.file.Memory
	if m == nil || !validState(m.Actor, len(f.Actor.Hidden.Weight)/3) || !validState(m.Value, len(f.Value.Hidden.Weight)/3) {
		return nil, 0, nil, fmt.Errorf("invalid recurrent sample state")
	}
	x, e := FeaturesForVersion(o, p.file.Features)
	if e != nil {
		return nil, 0, nil, e
	}
	a, ah := recurrentRaw(p.file.Actor, f.Actor, x, m.Actor)
	v, vh := recurrentRaw(p.file.Value, f.Value, x, m.Value)
	for _, y := range append(a, v...) {
		if math.IsNaN(y) || math.IsInf(y, 0) {
			return nil, 0, nil, fmt.Errorf("nonfinite recurrent inference")
		}
	}
	return a, v[0], &MemorySample{Actor: ah, Value: vh}, nil
}

// VerifyMemory replays every provider call, including calls excluded from loss.
// Review is pure; this method is the explicit offline sequence cursor.
func (p *PPO) VerifyMemory(o Observation, s Sample) error {
	if !reflect.DeepEqual(p.beforeMemory(o), s.Memory) {
		return fmt.Errorf("recurrent state/reset chain differs at frame %d", o.Identity.Frame)
	}
	if !p.IsRecurrent() {
		return nil
	}
	_, _, after, e := p.memoryRaw(o, s.Memory)
	if e != nil {
		return e
	}
	p.memory = memoryState{identity: o.Identity, actor: after.Actor, value: after.Value, position: after.Position, valid: true}
	return nil
}
func (p *PPO) ValueAfter(o Observation, s Sample, next Observation) (float64, error) {
	if !p.IsRecurrent() {
		return p.Value(next)
	}
	_, _, after, e := p.memoryRaw(o, s.Memory)
	if e != nil {
		return 0, e
	}
	if !SameLife(o.Identity, next.Identity) || next.Identity.Frame != o.Identity.Frame+1 {
		if p.file.Attention != nil {
			after = &MemorySample{Actor: []float64{}, Value: []float64{}, Reset: true}
		} else {
			after = &MemorySample{Actor: make([]float64, len(after.Actor)), Value: make([]float64, len(after.Value)), Reset: true}
		}
	}
	_, v, _, e := p.memoryRaw(next, after)
	return v, e
}
