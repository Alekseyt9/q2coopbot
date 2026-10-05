package policy

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
)

func writePPO(t *testing.T, seed int64) *PPO {
	t.Helper()
	f := fixtureMLP()
	v := fixtureMLP()
	v.Layers[2].Weight = v.Layers[2].Weight[:1]
	v.Layers[2].Bias = []float64{.5}
	b, e := json.Marshal(PPOFile{Kind: PPOKind, Features: FeatureVersion, Actor: f.Layers, Value: v.Layers, LogStd: [4]float64{-2, -2, -5, -5}, SamplingSeed: seed})
	if e != nil {
		t.Fatal(e)
	}
	path := filepath.Join(t.TempDir(), "ppo.json")
	if e = os.WriteFile(path, b, 0600); e != nil {
		t.Fatal(e)
	}
	p, e := LoadPPO(path)
	if e != nil {
		t.Fatal(e)
	}
	return p
}
func TestPPOReproducibleSamplesAndReview(t *testing.T) {
	a, b, c := writePPO(t, 10), writePPO(t, 10), writePPO(t, 11)
	if a.Version() != c.Version() {
		t.Fatal("sampling seed changed weight version")
	}
	o := testObservation()
	different := false
	for i := 0; i < 20; i++ {
		x, e := a.Decide(o)
		if e != nil {
			t.Fatal(e)
		}
		y, _ := b.Decide(o)
		z, _ := c.Decide(o)
		if x != y {
			t.Fatal("same stream not reproducible")
		}
		different = different || x != z
		s := *a.LastSample()
		review, lp, v, e := a.Review(o, s)
		if e != nil || review != x || lp != s.LogProbability || v != s.Value || math.IsNaN(lp) {
			t.Fatal("sample review failed")
		}
		s.Version = "wrong"
		if _, _, _, e = a.Review(o, s); e == nil {
			t.Fatal("accepted stale weights")
		}
	}
	if !different {
		t.Fatal("independent seeds reused RNG")
	}
}

func TestPPOCapacityWidthsAndBounds(t *testing.T) {
	for _, width := range []int{64, 128, 256, 257} {
		makeLayers := func(outputs int) []DenseLayer {
			n := 386
			var ls []DenseLayer
			for _, size := range []int{width, width, outputs} {
				l := DenseLayer{Bias: make([]float64, size), Weight: make([][]float64, size)}
				for i := range l.Weight {
					l.Weight[i] = make([]float64, n)
					for j := range l.Weight[i] {
						l.Weight[i][j] = math.Pi / 1000
					}
				}
				ls = append(ls, l)
				n = size
			}
			return ls
		}
		f := PPOFile{Kind: PPOKind, Features: FeatureVersion, Actor: makeLayers(8), Value: makeLayers(1), LogStd: [4]float64{-2, -2, -5, -5}}
		b, err := json.Marshal(f)
		if err != nil {
			t.Fatal(err)
		}
		if width == 256 && len(b) <= 2*1024*1024 {
			t.Fatal("fixture must exercise former size limit")
		}
		path := filepath.Join(t.TempDir(), "capacity.json")
		if err := os.WriteFile(path, b, 0600); err != nil {
			t.Fatal(err)
		}
		p, err := LoadPPO(path)
		if width == 257 {
			if err == nil {
				t.Fatal("accepted over-width model")
			}
			continue
		}
		if err != nil {
			t.Fatalf("width %d: %v", width, err)
		}
		o := testObservation()
		a, err := p.Decide(o)
		if err != nil {
			t.Fatal(err)
		}
		r, lp, v, err := p.Review(o, *p.LastSample())
		if err != nil || r != a || math.IsNaN(lp) || math.IsInf(lp, 0) || math.IsNaN(v) || math.IsInf(v, 0) {
			t.Fatal("wide sample review failed")
		}
	}
}
