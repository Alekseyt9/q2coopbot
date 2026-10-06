package policy

import (
	"fmt"
	"math"
)

const EntityAttentionVersion = "combat_entity_attention_v1"

type EntityAttentionCell struct {
	Token     DenseLayer    `json:"token"`
	Attention AttentionCell `json:"attention"`
}
type EntityAttentionFile struct {
	Version string              `json:"version"`
	Heads   int                 `json:"heads"`
	Actor   EntityAttentionCell `json:"actor"`
	Value   EntityAttentionCell `json:"value"`
}

func validateEntityAttention(f *EntityAttentionFile, p PPOFile) error {
	if f == nil {
		return nil
	}
	if p.Memory != nil || p.Attention != nil || p.Features != TypedFeatureVersion || f.Version != EntityAttentionVersion || f.Heads < 1 || f.Heads > 8 {
		return fmt.Errorf("invalid entity attention architecture")
	}
	for i, c := range []EntityAttentionCell{f.Actor, f.Value} {
		base := p.Actor
		outputs := 8
		if i == 1 {
			base = p.Value
			outputs = 1
		}
		w := len(base[1].Bias)
		a := c.Attention
		if w%f.Heads != 0 || !validMatrix(c.Token, w, 68) || !validMatrix(a.Query, w, w) || !validMatrix(a.Key, w, w) || !validMatrix(a.Value, w, w) || !validMatrix(a.Output, w, w) || !validMatrix(a.Residual, outputs, w) {
			return fmt.Errorf("invalid entity attention shape/weights")
		}
	}
	return nil
}

// Existing v4 values only; six role indicators add structure, not world truth.
func entityTokens(x []float64) ([][]float64, error) {
	if len(x) != 810 {
		return nil, fmt.Errorf("entity tokens require v4/810")
	}
	tokens := [][]float64{}
	put := func(role int, data []float64, valid bool) {
		if !valid {
			return
		}
		t := make([]float64, 68)
		copy(t, data)
		t[62+role] = 1
		tokens = append(tokens, t)
	}
	put(0, append(append([]float64{}, x[:25]...), x[786:810]...), true)
	for i := 0; i < 8; i++ {
		j := 73 + 12*i
		d := append([]float64{}, x[j:j+12]...)
		d = append(d, x[386+5*i:391+5*i]...)
		d = append(d, x[426+5*i:431+5*i]...)
		d = append(d, x[466+40*i:506+40*i]...)
		put(1, d, x[j] != 0)
	}
	for i := 0; i < 4; i++ {
		j := 170 + 12*i
		put(2, x[j:j+12], x[j] != 0)
	}
	for i := 0; i < 8; i++ {
		j := 25 + 6*i
		put(3, x[j:j+6], x[j] != 0)
	}
	for role, start := range []int{223, 260} {
		for i := 0; i < 4; i++ {
			j := start + 9*i
			put(4+role, x[j:j+9], x[j] != 0)
		}
	}
	return tokens, nil
}
func entityForward(base []DenseLayer, c EntityAttentionCell, x []float64, tokens [][]float64, heads int) []float64 {
	encoded := x
	for _, l := range base[:2] {
		encoded = affine(l, encoded)
		for i := range encoded {
			encoded[i] = math.Max(0, encoded[i])
		}
	}
	q := affine(c.Attention.Query, encoded)
	keys, values := make([][]float64, len(tokens)), make([][]float64, len(tokens))
	for i, t := range tokens {
		e := affine(c.Token, t)
		for j := range e {
			e[j] = math.Max(0, e[j])
		}
		keys[i] = affine(c.Attention.Key, e)
		values[i] = affine(c.Attention.Value, e)
	}
	width := len(q)
	size := width / heads
	attended := make([]float64, width)
	for head := 0; head < heads; head++ {
		scores := make([]float64, len(tokens))
		max := -math.MaxFloat64
		for t := range tokens {
			for j := head * size; j < (head+1)*size; j++ {
				scores[t] += q[j] * keys[t][j] / math.Sqrt(float64(size))
			}
			max = math.Max(max, scores[t])
		}
		sum := 0.
		for t := range scores {
			scores[t] = math.Exp(scores[t] - max)
			sum += scores[t]
		}
		for t := range scores {
			for j := head * size; j < (head+1)*size; j++ {
				attended[j] += scores[t] / sum * values[t][j]
			}
		}
	}
	y := affine(base[2], encoded)
	delta := affine(c.Attention.Residual, affine(c.Attention.Output, attended))
	for i := range y {
		y[i] += delta[i]
	}
	return y
}
func (p *PPO) entityRaw(o Observation) ([]float64, float64, *MemorySample, error) {
	x, e := FeaturesForVersion(o, p.file.Features)
	if e != nil {
		return nil, 0, nil, e
	}
	tokens, e := entityTokens(x)
	if e != nil {
		return nil, 0, nil, e
	}
	f := p.file.EntityAttention
	a := entityForward(p.file.Actor, f.Actor, x, tokens, f.Heads)
	v := entityForward(p.file.Value, f.Value, x, tokens, f.Heads)
	for _, y := range append(a, v...) {
		if math.IsNaN(y) || math.IsInf(y, 0) {
			return nil, 0, nil, fmt.Errorf("nonfinite entity attention inference")
		}
	}
	return a, v[0], nil, nil
}
