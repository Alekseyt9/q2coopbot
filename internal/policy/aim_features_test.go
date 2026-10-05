package policy

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"q2coopbot/internal/quake"
	"reflect"
	"testing"
)

func TestAimFeaturesGeometryAndMasks(t *testing.T) {
	clear := true
	o := testObservation()
	o.ViewAngles = [3]int16{}
	o.Enemies = []Enemy{{Relative: quake.Vec3{22, 0, 0}, Distance: 22, ClearShot: &clear}}
	legacy, _ := Features(o)
	x, err := FeaturesForVersion(o, AimFeatureVersion)
	if err != nil || len(x) != 426 || !reflect.DeepEqual(x[:386], legacy) {
		t.Fatalf("prefix/version: %v", err)
	}
	want := []float64{1, 0, 1, math.Sqrt(.5), math.Sqrt(.5)}
	for i, v := range want {
		if math.Abs(x[386+i]-v) > 1e-9 {
			t.Fatalf("feature %d: %g want %g", i, x[386+i], v)
		}
	}
	o.Ducked = true
	o.Enemies[0].Relative = quake.Vec3{2, 0, 0}
	x, _ = FeaturesForVersion(o, AimFeatureVersion)
	if math.Abs(x[389]+math.Sqrt(.5)) > 1e-9 {
		t.Fatal("crouch/pitch sign")
	}
	o.Enemies[0].Relative = quake.Vec3{0, 0, -2}
	x, _ = FeaturesForVersion(o, AimFeatureVersion)
	for _, v := range x[386:] {
		if v != 0 {
			t.Fatal("degenerate target must be masked")
		}
	}
	o.Enemies[0].Relative = quake.Vec3{2, 0, 0}
	clear = false
	x, _ = FeaturesForVersion(o, AimFeatureVersion)
	for _, v := range x[386:] {
		if v != 0 {
			t.Fatal("occluded target must be masked")
		}
	}
	if _, err = FeaturesForVersion(o, "unknown"); err == nil {
		t.Fatal("unknown feature version")
	}
}

func TestBBoxFeaturesObservedBoundsAndLegacyPrefix(t *testing.T) {
	clear := true
	solid := uint16(8290)
	o := testObservation()
	o.ViewAngles = [3]int16{}
	o.Enemies = []Enemy{{Distance: 100, Relative: quake.Vec3{100, 0, 0}, Solid: &solid, ClearShot: &clear}}
	v2, _ := FeaturesForVersion(o, AimFeatureVersion)
	v3, err := FeaturesForVersion(o, BBoxFeatureVersion)
	if err != nil || len(v3) != 466 || !reflect.DeepEqual(v2, v3[:426]) {
		t.Fatal("v3 prefix", err)
	}
	want := []float64{1, 0, 1, 0, 1}
	for i, v := range want {
		if math.Abs(v3[426+i]-v) > 1e-9 {
			t.Fatal("upper body angles", v3[426:431])
		}
	}
	solid = 4194 // top zero, body aim -8; standing eye +22
	r, known, err := ObservedAimDirection(o, o.Enemies[0])
	if err != nil || !known || math.Abs(r[2]/r[0]+.3) > 1e-9 {
		t.Fatal("packed bbox clamp", r, err)
	}
	o.Enemies[0].Solid = nil
	v3, _ = FeaturesForVersion(o, BBoxFeatureVersion)
	for _, v := range v3[426:] {
		if v != 0 {
			t.Fatal("unknown bbox mask")
		}
	}
}

func TestAimFeaturesRotationAndIdentity(t *testing.T) {
	clear := true
	o := testObservation()
	o.ViewAngles = [3]int16{}
	o.Enemies = []Enemy{{Relative: quake.Vec3{100, 0, 22}, Distance: 100, ClearShot: &clear}}
	a, _ := FeaturesForVersion(o, AimFeatureVersion)
	o.ViewAngles[1] = 16384
	o.Enemies[0].Relative = quake.Vec3{0, 100, 22}
	o.Identity.Frame += 100
	o.Identity.Actor += 1
	b, _ := FeaturesForVersion(o, AimFeatureVersion)
	for i := 386; i < len(a); i++ {
		if math.Abs(a[i]-b[i]) > 1e-9 {
			t.Fatal("rotation/identity changed angle features")
		}
	}
	o.Enemies[0].Relative[0] = math.NaN()
	if _, err := FeaturesForVersion(o, AimFeatureVersion); err == nil {
		t.Fatal("nonfinite target accepted")
	}
}

func TestPPOAimVersionRoutesFeaturesAndPreservesZeroExtension(t *testing.T) {
	actor := fixtureMLP()
	value := fixtureMLP()
	value.Layers[2].Weight = value.Layers[2].Weight[:1]
	value.Layers[2].Bias = value.Layers[2].Bias[:1]
	f := PPOFile{Kind: PPOKind, Features: AimFeatureVersion, Actor: actor.Layers, Value: value.Layers, LogStd: [4]float64{-2, -2, -5, -5}, Deterministic: true}
	for _, layers := range [][]DenseLayer{f.Actor, f.Value} {
		for i := range layers[0].Weight {
			layers[0].Weight[i] = append(layers[0].Weight[i], make([]float64, 40)...)
		}
	}
	p := filepath.Join(t.TempDir(), "weights.json")
	data, _ := json.Marshal(f)
	os.WriteFile(p, data, 0600)
	m, err := LoadPPO(p)
	if err != nil {
		t.Fatal(err)
	}
	o := testObservation()
	x, _ := m.actor.Raw(o)
	if !reflect.DeepEqual(x, actor.Layers[2].Bias) {
		t.Fatal("zero extension changed fixture outputs")
	}
	if _, err = m.Decide(o); err != nil {
		t.Fatal(err)
	}
	if _, err = m.Value(o); err != nil {
		t.Fatal(err)
	}
	f.Features = FeatureVersion
	data, _ = json.Marshal(f)
	os.WriteFile(p, data, 0600)
	if _, err = LoadPPO(p); err == nil {
		t.Fatal("version/width mismatch accepted")
	}
}
