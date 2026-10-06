package policy

import (
	"math"
	"reflect"
	"testing"
)

func gruMatrix(rows, cols int) DenseLayer {
	l := DenseLayer{Bias: make([]float64, rows), Weight: make([][]float64, rows)}
	for i := range l.Weight {
		l.Weight[i] = make([]float64, cols)
	}
	return l
}
func memoryFixture(p *PPO) {
	cell := func(base []DenseLayer) GRUCell {
		c := GRUCell{Input: gruMatrix(3, len(base[1].Bias)), Hidden: gruMatrix(3, 1), Output: gruMatrix(len(base[2].Bias), 1)}
		c.Input.Bias[2] = .4
		c.Hidden.Weight[2][0] = .5
		return c
	}
	p.file.Memory = &GRUFile{Version: GRUVersion, Actor: cell(p.file.Actor), Value: cell(p.file.Value)}
}
func TestGRUTorchCandidateEquation(t *testing.T) {
	c := GRUCell{Input: gruMatrix(3, 1), Hidden: gruMatrix(3, 1), Output: gruMatrix(1, 1)}
	c.Input.Bias = []float64{.3, -.7, .2}
	c.Hidden.Bias = []float64{-.1, .4, .9}
	c.Hidden.Weight[2][0] = .5
	c.Output.Weight[0][0] = 1
	y, h := gruForward(c, []float64{0}, []float64{.4})
	r, z := 1/(1+math.Exp(-.2)), 1/(1+math.Exp(.3))
	want := (1-z)*math.Tanh(.2+r*(.9+.5*.4)) + z*.4
	if math.Abs(h[0]-want) > 1e-14 || y[0] != h[0] {
		t.Fatal(y, h, want)
	}
}
func TestGRUZeroResidualAndMemoryBoundaries(t *testing.T) {
	p := writePPO(t, 42)
	q := writePPO(t, 42)
	memoryFixture(q)
	o := testObservation()
	for i := 0; i < 5; i++ {
		o.Identity.Frame++
		a, e := p.Decide(o)
		if e != nil {
			t.Fatal(e)
		}
		b, e := q.Decide(o)
		if e != nil || a != b {
			t.Fatal("zero residual changes action", e)
		}
		before := *q.LastSample()
		duplicate, e := q.Decide(o)
		if e != nil || duplicate != b || !reflect.DeepEqual(before, *q.LastSample()) {
			t.Fatal("duplicate advances memory/RNG")
		}
		if i > 0 && q.LastSample().Memory.Reset {
			t.Fatal("consecutive frame reset")
		}
	}
	baseline := q.memory.identity
	changes := []Identity{baseline, baseline, baseline, baseline, baseline, baseline, baseline}
	changes[0].Life++
	changes[1].Map = "other"
	changes[2].Connection++
	changes[3].Actor++
	changes[4].Spawncount++
	changes[5].Frame += 2
	changes[6].Frame--
	for _, id := range changes {
		o.Identity = id
		m := q.beforeMemory(o)
		if !m.Reset || m.Actor[0] != 0 || m.Value[0] != 0 {
			t.Fatal("memory crossed boundary", id)
		}
	}
}
func TestGRUReplayRejectsAlteredHiddenStateAndBootstrapIsPure(t *testing.T) {
	p := writePPO(t, 9)
	q := writePPO(t, 9)
	memoryFixture(p)
	memoryFixture(q)
	o := testObservation()
	for i := 0; i < 3; i++ {
		o.Identity.Frame++
		a, e := p.Decide(o)
		if e != nil {
			t.Fatal(e)
		}
		s := *p.LastSample()
		if e = q.VerifyMemory(o, s); e != nil {
			t.Fatal(e)
		}
		review, lp, value, e := q.Review(o, s)
		if e != nil || review != a || lp != s.LogProbability || value != s.Value {
			t.Fatal("replay mismatch", e)
		}
		next := o
		next.Identity.Frame++
		state := q.memory
		if _, e = q.ValueAfter(o, s, next); e != nil {
			t.Fatal(e)
		}
		if !reflect.DeepEqual(state, q.memory) {
			t.Fatal("bootstrap mutated memory")
		}
	}
	o.Identity.Frame++
	_, _ = p.Decide(o)
	s := *p.LastSample()
	altered := *s.Memory
	altered.Actor = append([]float64{}, altered.Actor...)
	altered.Actor[0] += .1
	s.Memory = &altered
	if q.VerifyMemory(o, s) == nil {
		t.Fatal("altered hidden accepted")
	}
	s.Memory.Actor = []float64{math.NaN()}
	if _, _, _, e := q.Review(o, s); e == nil {
		t.Fatal("nonfinite hidden accepted")
	}
}
