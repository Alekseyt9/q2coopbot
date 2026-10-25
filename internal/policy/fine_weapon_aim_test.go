package policy

import (
	"math"
	"testing"
)

// Metadata and gating only. Neural numerical checks execute on CUDA/native.
func TestFineWeaponAimContract(t *testing.T) {
	makeFile := func() PPOFile {
		s := &SpatialAimFile{Version: FineWeaponAimVersion}
		width := FineWeaponAimInputWidth
		for _, output := range []int{32, 16, 2} {
			l := DenseLayer{Weight: make([][]float64, output), Bias: make([]float64, output)}
			for i := range l.Weight {
				l.Weight[i] = make([]float64, width)
			}
			s.Layers = append(s.Layers, l)
			width = output
		}
		return PPOFile{Features: EquippedWeaponFeatureVersion, SpatialAim: &SpatialAimFile{},
			AimModeHead: PrecisionHeadVersion, SampledMotionAim: &SpatialAimFile{Version: SampledMotionAimVersion}, FineWeaponAim: s}
	}
	if err := validateFineWeaponAim(makeFile()); err != nil {
		t.Fatal(err)
	}
	for _, mutate := range []func(*PPOFile){
		func(f *PPOFile) { f.FineWeaponAim.Version = "unknown" },
		func(f *PPOFile) { f.SampledMotionAim = nil },
		func(f *PPOFile) { f.SampledMotionAim.Version = SampledVerticalAimVersion },
		func(f *PPOFile) { f.SpatialAim = nil },
		func(f *PPOFile) { f.AimModeHead = "" },
		func(f *PPOFile) { f.FineWeaponAim.Layers[0].Weight[0] = nil },
		func(f *PPOFile) { f.FineWeaponAim.Layers[2].Bias[0] = math.NaN() },
	} {
		f := makeFile()
		mutate(&f)
		if validateFineWeaponAim(f) == nil {
			t.Fatal("accepted invalid fine-weapon contract")
		}
	}
}

func TestFineWeaponAimObservedGate(t *testing.T) {
	x := make([]float64, EquippedWeaponFeatureWidth)
	x[0], x[1123], x[73+12*2], x[426+5*2] = 1, 1, 1, 1
	if !fineWeaponAimActive(x, 3, 1) {
		t.Fatal("available observed Shotgun fine target rejected")
	}
	for _, pair := range [][2]int{{3, 0}, {0, 1}, {1, 1}, {9, 1}} {
		if fineWeaponAimActive(x, pair[0], pair[1]) {
			t.Fatal("gate active outside chosen fine target", pair)
		}
	}
	for _, column := range []int{0, 1123, 73 + 12*2, 426 + 5*2} {
		x[column] = 0
		if fineWeaponAimActive(x, 3, 1) {
			t.Fatal("gate invented unavailable fact", column)
		}
		x[column] = 1
	}
}
