package policy

import (
	"math"
	"testing"
)

// Contract-only validation; neural output/replay verification runs on CUDA.
func TestWeaponStateResidualContract(t *testing.T) {
	makeFile := func() PPOFile {
		layers := make([]DenseLayer, 3)
		width := WeaponStateResidualInputWidth
		for i, out := range []int{64, 32, 12} {
			layers[i].Weight = make([][]float64, out)
			layers[i].Bias = make([]float64, out)
			for j := range layers[i].Weight {
				layers[i].Weight[j] = make([]float64, width)
			}
			width = out
		}
		return PPOFile{Features: EquippedWeaponFeatureVersion, AimModeHead: PrecisionHeadVersion, WeaponHead: WeaponHeadVersion, TargetHead: TargetHeadVersion, SpatialAim: &SpatialAimFile{}, WeaponStateResidual: &SpatialAimFile{Version: WeaponStateResidualVersion, Layers: layers}}
	}
	if err := validateWeaponStateResidual(PPOFile{}); err != nil {
		t.Fatal(err)
	}
	if err := validateWeaponStateResidual(makeFile()); err != nil {
		t.Fatal(err)
	}
	for name, mutate := range map[string]func(*PPOFile){
		"legacy features":   func(f *PPOFile) { f.Features = ThreatFeatureVersion },
		"private prototype": func(f *PPOFile) { f.WeaponStateResidual.Version = "combat_observed_weapon_state_residual_prototype_v1" },
		"missing precision": func(f *PPOFile) { f.AimModeHead = "" },
		"missing spatial":   func(f *PPOFile) { f.SpatialAim = nil },
		"missing weapon":    func(f *PPOFile) { f.WeaponHead = "" },
		"missing target":    func(f *PPOFile) { f.TargetHead = "" },
		"width":             func(f *PPOFile) { f.WeaponStateResidual.Layers[0].Weight[0] = nil },
		"nonfinite":         func(f *PPOFile) { f.WeaponStateResidual.Layers[2].Bias[0] = math.NaN() },
	} {
		t.Run(name, func(t *testing.T) {
			f := makeFile()
			mutate(&f)
			if validateWeaponStateResidual(f) == nil {
				t.Fatal("invalid contract accepted")
			}
		})
	}
}
