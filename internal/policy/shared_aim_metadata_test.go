package policy

import (
	"math"
	"testing"
)

// Contract validation only; learned forward passes are checked on CUDA/live.
func TestSharedAimScorerMetadata(t *testing.T) {
	valid := func() *SharedAimScorerFile {
		s := &SharedAimScorerFile{Kind: "shared_candidate_causal191_v1", ScoreClip: 20, BetaBounds: []float64{-20, 20}, Beta: 2}
		for i, width := range []int{191, 16} {
			output := []int{16, 1}[i]
			l := DenseLayer{Weight: make([][]float64, output), Bias: make([]float64, output)}
			for r := range l.Weight {
				l.Weight[r] = make([]float64, width)
			}
			s.Layers = append(s.Layers, l)
		}
		return s
	}
	if err := validateSharedAimScorer(valid()); err != nil {
		t.Fatal(err)
	}
	g := valid()
	g.Kind = GeometryDeltaAimKind
	g.Layers = []DenseLayer{{Weight: make([][]float64, 8), Bias: make([]float64, 8)}, {Weight: [][]float64{make([]float64, 8)}, Bias: make([]float64, 1)}}
	for r := range g.Layers[0].Weight {
		g.Layers[0].Weight[r] = make([]float64, 31)
	}
	if err := validateSharedAimScorer(g); err != nil {
		t.Fatal(err)
	}
	g.Layers[0].Weight[0] = make([]float64, 191)
	if validateSharedAimScorer(g) == nil {
		t.Fatal("legacy feature width accepted for geometry delta scorer")
	}
	for _, mutate := range []func(*SharedAimScorerFile){
		func(s *SharedAimScorerFile) { s.Beta = math.NaN() },
		func(s *SharedAimScorerFile) { s.Beta = 21 },
		func(s *SharedAimScorerFile) { s.ScoreClip = 19 },
		func(s *SharedAimScorerFile) { s.Layers[0].Weight[0] = nil },
		func(s *SharedAimScorerFile) { s.Layers[1].Bias[0] = math.Inf(1) },
		func(s *SharedAimScorerFile) { s.Kind = "unknown" },
	} {
		s := valid()
		mutate(s)
		if validateSharedAimScorer(s) == nil {
			t.Fatal("accepted invalid shared scorer metadata")
		}
	}
}
