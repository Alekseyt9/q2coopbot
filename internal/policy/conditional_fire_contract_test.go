package policy

import (
	"math"
	"testing"
)

// Layout/validation only. No neural network is executed on the CPU.
func TestConditionalFireContractAndInputs(t *testing.T) {
	spec := &SpatialAimFile{Version: ConditionalFireVersion}
	width := ConditionalFireInputWidth
	for _, out := range []int{32, 16, 1} {
		layer := DenseLayer{Bias: make([]float64, out), Weight: make([][]float64, out)}
		for i := range layer.Weight {
			layer.Weight[i] = make([]float64, width)
		}
		spec.Layers = append(spec.Layers, layer)
		width = out
	}
	if err := validateConditionalFireSpec(spec); err != nil {
		t.Fatal(err)
	}
	header := PPOFile{Features: EquippedWeaponFeatureVersion, AimModeHead: PrecisionHeadVersion,
		SpatialAim: &SpatialAimFile{}, SampledMotionAim: &SpatialAimFile{}, ConditionalFire: spec}
	if err := validateConditionalFire(header); err != nil {
		t.Fatal(err)
	}
	for _, change := range []func(*PPOFile){
		func(f *PPOFile) { f.Features = FeatureVersion },
		func(f *PPOFile) { f.AimModeHead = "" },
		func(f *PPOFile) { f.SpatialAim = nil },
		func(f *PPOFile) { f.SampledMotionAim = nil },
		func(f *PPOFile) { f.EntityAttention = &EntityAttentionFile{} },
	} {
		bad := header
		change(&bad)
		if validateConditionalFire(bad) == nil {
			t.Fatal("accepted unsupported conditional fire header")
		}
	}
	legacy := &PPO{}
	if value, err := legacy.conditionalFireLogit(Observation{}, -.7, Sample{}); err != nil || value != -.7 {
		t.Fatal("legacy fire logit changed")
	}
	features := make([]float64, 1147)
	for i := 0; i < 73; i++ {
		features[i] = float64(i) / 100
	}
	for i := 1121; i < 1147; i++ {
		features[i] = float64(i)
	}
	features[WeaponMaskOffset] = 1
	features[WeaponMaskOffset+4] = 1
	features[431] = 1 // target category2: bbox availability of slot1
	features[85], features[506] = 7, 9
	sample := Sample{Target: 2, AimMode: 1, Vertical: 1, Weapon: 4, Latent: [4]float64{.25, 0, .5, 0}}
	x, err := conditionalFireInputs(features, sample)
	if err != nil {
		t.Fatal(err)
	}
	if len(x) != 181 || x[73] != 1121 || x[98] != 1146 || x[99] != 7 || x[111] != 9 || x[153] != 1 || x[161] != 1 || x[167] != 1 || x[173] != 1 || math.Abs(x[164]-math.Tanh(.5)/12) > 1e-15 {
		t.Fatal("conditional fire input layout")
	}
	sample.Target = 0
	x, err = conditionalFireInputs(features, sample)
	if err != nil {
		t.Fatal(err)
	}
	for _, value := range x[99:151] {
		if value != 0 {
			t.Fatal("none target retained another enemy")
		}
	}
	sample.Target = 1
	if _, err := conditionalFireInputs(features, sample); err == nil {
		t.Fatal("accepted unavailable target")
	}
	sample.Target = 0
	sample.Latent[1] = math.NaN()
	if _, err := conditionalFireInputs(features, sample); err == nil {
		t.Fatal("accepted nonfinite action")
	}
	spec.Layers[0].Weight[0][0] = math.NaN()
	if validateConditionalFireSpec(spec) == nil {
		t.Fatal("accepted nonfinite parameter")
	}
	if err := validateConditionalFireSpec(nil); err != nil {
		t.Fatal(err)
	}
}
