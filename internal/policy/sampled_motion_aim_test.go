package policy

import (
	"math"
	"testing"
)

// Contract validation only: numeric NN execution is tested on CUDA and native runs.
func TestSampledMotionAimContract(t *testing.T) {
	makeFile := func() PPOFile {
		spec := &SpatialAimFile{Version: SampledMotionAimVersion}
		width := SampledMotionAimInputWidth
		for _, output := range []int{64, 32, 2} {
			layer := DenseLayer{Weight: make([][]float64, output), Bias: make([]float64, output)}
			for r := range layer.Weight {
				layer.Weight[r] = make([]float64, width)
			}
			spec.Layers = append(spec.Layers, layer)
			width = output
		}
		return PPOFile{Features: EquippedWeaponFeatureVersion, SpatialAim: &SpatialAimFile{}, AimModeHead: PrecisionHeadVersion, SampledMotionAim: spec}
	}
	if err := validateSampledMotionAim(makeFile()); err != nil {
		t.Fatal(err)
	}
	for _, mutate := range []func(*PPOFile){
		func(f *PPOFile) { f.Features = TargetFeatureVersion },
		func(f *PPOFile) { f.SpatialAim = nil },
		func(f *PPOFile) { f.AimModeHead = "" },
		func(f *PPOFile) { f.SampledMotionAim.Version = "unknown" },
		func(f *PPOFile) { f.SampledMotionAim.Layers = f.SampledMotionAim.Layers[:2] },
		func(f *PPOFile) { f.SampledMotionAim.Layers[0].Weight[0] = []float64{0} },
		func(f *PPOFile) { f.SampledMotionAim.Layers[2].Bias = []float64{0} },
		func(f *PPOFile) { f.SampledMotionAim.Layers[0].Weight[0][0] = math.NaN() },
		func(f *PPOFile) { f.SampledMotionAim.Layers[0].Bias[0] = 10001 },
	} {
		f := makeFile()
		mutate(&f)
		if validateSampledMotionAim(f) == nil {
			t.Fatal("accepted incompatible sampled-motion contract")
		}
	}
	if err := validateSampledMotionAim(PPOFile{}); err != nil {
		t.Fatal("legacy policy must remain valid", err)
	}
}
