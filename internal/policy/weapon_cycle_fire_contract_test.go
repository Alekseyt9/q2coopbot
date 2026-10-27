package policy

import "testing"

// Contract/gate metadata only: numerical network verification runs on CUDA.
func TestWeaponCycleFireContract(t *testing.T) {
	features := make([]float64, EquippedWeaponFeatureWidth)
	features[0], features[1132], features[5] = 1, 1, 11.0/30
	if !weaponFireResidualActiveForVersion(features, WeaponCycleFireResidualVersion) {
		t.Fatal("observed live hand-grenade hold phase omitted")
	}
	for weapon := 0; weapon < 12; weapon++ {
		for frame := 0; frame < 30; frame++ {
			for i := 0; i < 12; i++ {
				features[1121+i] = 0
			}
			features[1121+weapon], features[5] = 1, float64(frame)/30
			if got := weaponFireResidualActiveForVersion(features, WeaponCycleFireResidualVersion); got != (weapon == 11 && frame == 11) {
				t.Fatalf("gate for observed weapon %d frame %d: %v", weapon, frame, got)
			}
		}
	}
	features[0], features[1132], features[5] = 0, 1, 11.0/30
	if weaponFireResidualActiveForVersion(features, WeaponCycleFireResidualVersion) || weaponFireResidualActiveForVersion(features, "unknown") {
		t.Fatal("dead/unknown residual admitted")
	}
	spec := &SpatialAimFile{Version: WeaponCycleFireResidualVersion}
	width := ConditionalFireContextInputWidth
	for _, output := range []int{32, 16, 1} {
		layer := DenseLayer{Bias: make([]float64, output), Weight: make([][]float64, output)}
		for i := range layer.Weight {
			layer.Weight[i] = make([]float64, width)
		}
		spec.Layers = append(spec.Layers, layer)
		width = output
	}
	f := PPOFile{ConditionalFire: &SpatialAimFile{}, WeaponFireResidual: spec}
	if err := validateWeaponFireResidual(f); err != nil {
		t.Fatal(err)
	}
	spec.Version = WeaponFireResidualVersion
	if validateWeaponFireResidual(f) == nil {
		t.Fatal("267-column cycle weights accepted as old SG residual")
	}
}
