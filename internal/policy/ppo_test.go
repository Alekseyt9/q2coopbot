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
