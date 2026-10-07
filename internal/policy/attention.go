package policy

import (
	"fmt"
	"math"
)

const AttentionVersion = "combat_causal_attention_v1"

type AttentionCell struct {
	Query    DenseLayer `json:"query"`
	Key      DenseLayer `json:"key"`
	Value    DenseLayer `json:"value"`
	Output   DenseLayer `json:"output"`
	Residual DenseLayer `json:"residual"`
}
type AttentionFile struct {
	Version string        `json:"version"`
	Heads   int           `json:"heads"`
	Window  int           `json:"window"`
	Actor   AttentionCell `json:"actor"`
	Value   AttentionCell `json:"value"`
}

func validateAttention(f *AttentionFile, g *GRUFile, actor, value []DenseLayer) error {
	if f == nil {
		return nil
	}
	if g != nil || f.Version != AttentionVersion || f.Heads < 1 || f.Heads > 8 || f.Window < 2 || f.Window > 64 {
		return fmt.Errorf("invalid attention architecture")
	}
	for i, c := range []AttentionCell{f.Actor, f.Value} {
		base := actor
		out := len(actor[len(actor)-1].Bias)
		if i == 1 {
			base = value
			out = 1
		}
		width := len(base[1].Bias)
		if width%f.Heads != 0 || !validMatrix(c.Query, width, width) || !validMatrix(c.Key, width, width) || !validMatrix(c.Value, width, width) || !validMatrix(c.Output, width, width) || !validMatrix(c.Residual, out, width) {
			return fmt.Errorf("invalid attention shape/weights")
		}
	}
	return nil
}
func attentionForward(c AttentionCell, current, previous []float64, heads, window, position int) ([]float64, []float64) {
	width := len(current)
	embedded := append([]float64{}, current...)
	for i := range embedded {
		angle := float64(position) * math.Exp(-math.Log(10000)*float64(2*(i/2))/float64(width))
		if i%2 == 0 {
			embedded[i] += math.Sin(angle)
		} else {
			embedded[i] += math.Cos(angle)
		}
	}
	tokens := append(append([]float64{}, previous...), embedded...)
	if len(tokens) > window*width {
		tokens = tokens[len(tokens)-window*width:]
	}
	count := len(tokens) / width
	q := affine(c.Query, embedded)
	keys, values := make([][]float64, count), make([][]float64, count)
	for i := 0; i < count; i++ {
		keys[i] = affine(c.Key, tokens[i*width:(i+1)*width])
		values[i] = affine(c.Value, tokens[i*width:(i+1)*width])
	}
	attended := make([]float64, width)
	size := width / heads
	for head := 0; head < heads; head++ {
		logits := make([]float64, count)
		max := -math.MaxFloat64
		for t := 0; t < count; t++ {
			for j := head * size; j < (head+1)*size; j++ {
				logits[t] += q[j] * keys[t][j] / math.Sqrt(float64(size))
			}
			max = math.Max(max, logits[t])
		}
		sum := 0.
		for t := range logits {
			logits[t] = math.Exp(logits[t] - max)
			sum += logits[t]
		}
		for t := range logits {
			for j := head * size; j < (head+1)*size; j++ {
				attended[j] += logits[t] / sum * values[t][j]
			}
		}
	}
	keep := tokens
	if len(keep) > (window-1)*width {
		keep = keep[len(keep)-(window-1)*width:]
	}
	return affine(c.Residual, affine(c.Output, attended)), append([]float64{}, keep...)
}
func validAttentionState(x []float64, width, window int) bool {
	if len(x)%width != 0 || len(x) > (window-1)*width {
		return false
	}
	for _, v := range x {
		if math.IsNaN(v) || math.IsInf(v, 0) || math.Abs(v) > 1e6 {
			return false
		}
	}
	return true
}
func (p *PPO) attentionRaw(o Observation, m *MemorySample) ([]float64, float64, *MemorySample, error) {
	f := p.file.Attention
	aw, vw := len(p.file.Actor[1].Bias), len(p.file.Value[1].Bias)
	if m == nil || m.Position < 0 || m.Position > 1000000 || len(m.Actor) != min(m.Position, f.Window-1)*aw || len(m.Value) != min(m.Position, f.Window-1)*vw || !validAttentionState(m.Actor, aw, f.Window) || !validAttentionState(m.Value, vw, f.Window) {
		return nil, 0, nil, fmt.Errorf("invalid attention sample state")
	}
	x, e := FeaturesForVersion(o, p.file.Features)
	if e != nil {
		return nil, 0, nil, e
	}
	run := func(base []DenseLayer, c AttentionCell, previous []float64) ([]float64, []float64) {
		encoded := x
		for _, l := range base[:2] {
			encoded = affine(l, encoded)
			for i := range encoded {
				encoded[i] = math.Max(0, encoded[i])
			}
		}
		y := affine(base[2], encoded)
		delta, next := attentionForward(c, encoded, previous, f.Heads, f.Window, m.Position)
		for i := range y {
			y[i] += delta[i]
		}
		return y, next
	}
	a, ah := run(p.file.Actor, f.Actor, m.Actor)
	v, vh := run(p.file.Value, f.Value, m.Value)
	for _, y := range append(a, v...) {
		if math.IsNaN(y) || math.IsInf(y, 0) {
			return nil, 0, nil, fmt.Errorf("nonfinite attention inference")
		}
	}
	return a, v[0], &MemorySample{Actor: ah, Value: vh, Position: m.Position + 1}, nil
}
