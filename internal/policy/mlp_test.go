package policy

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func fixtureMLP() MLPFile {
	x, _ := Features(Observation{})
	f := MLPFile{Kind: MLPKind, Features: FeatureVersion}
	n := len(x)
	for _, width := range []int{2, 2, 8} {
		l := DenseLayer{Bias: make([]float64, width), Weight: make([][]float64, width)}
		for i := range l.Weight {
			l.Weight[i] = make([]float64, n)
		}
		f.Layers = append(f.Layers, l)
		n = width
	}
	f.Layers[2].Bias = []float64{1, -1, .1, -.1, 2, 0, 3, 1}
	return f
}

func TestMLPRejectsMalformedFiles(t *testing.T) {
	for _, kind := range []string{"shape", "weight", "version", "trailing", "unknown"} {
		t.Run(kind, func(t *testing.T) {
			f := fixtureMLP()
			switch kind {
			case "shape":
				f.Layers[0].Weight[0] = nil
			case "weight":
				f.Layers[0].Weight[0][0] = 1e5
			case "version":
				f.Features = "unrecognized"
			}
			b, _ := json.Marshal(f)
			if kind == "trailing" {
				b = append(b, []byte(" {}")...)
			}
			if kind == "unknown" {
				b = append([]byte(`{"surprise":true,`), b[1:]...)
			}
			p := filepath.Join(t.TempDir(), "weights.json")
			if err := os.WriteFile(p, b, 0600); err != nil {
				t.Fatal(err)
			}
			if _, err := LoadMLP(p); err == nil {
				t.Fatal("accepted invalid model")
			}
		})
	}
}

func TestMLPActionAndObservationSafety(t *testing.T) {
	b, _ := json.Marshal(fixtureMLP())
	p := filepath.Join(t.TempDir(), "weights.json")
	if err := os.WriteFile(p, b, 0600); err != nil {
		t.Fatal(err)
	}
	m, err := LoadMLP(p)
	if err != nil {
		t.Fatal(err)
	}
	o := testObservation()
	a, err := m.Decide(o)
	if err != nil {
		t.Fatal(err)
	}
	if a.Identity != o.Identity || !a.Attack || a.Vertical != "jump" || a.Forward <= 0 || a.Side >= 0 || a.Weapon != "" {
		t.Fatalf("bad action: %+v", a)
	}
	o.Health = 0
	if _, err := m.Decide(o); err == nil {
		t.Fatal("accepted dead observation")
	}
	o.Health = 100
	o.Velocity[0] = math.NaN()
	if _, err := m.Raw(o); err == nil {
		t.Fatal("accepted nonfinite feature")
	}
}

func TestFeaturesExcludeIdentityAndPreserveUnknownMask(t *testing.T) {
	o := testObservation()
	a, err := Features(o)
	if err != nil {
		t.Fatal(err)
	}
	o.Identity.Frame += 99
	o.Identity.Actor += 10
	o.Identity.Map = "another-map"
	b, _ := Features(o)
	if !reflect.DeepEqual(a, b) {
		t.Fatal("identity leaked into features")
	}
	age := 0
	o.InventoryAgeFrames = &age
	c, _ := Features(o)
	if len(c) != len(a) || reflect.DeepEqual(a, c) {
		t.Fatal("missing and known zero indistinguishable")
	}
}
