package policy

import "testing"

// Metadata and applicability only; neural arithmetic is verified on CUDA.
func TestWeaponFireResidualContractAndObservedGate(t *testing.T) {
	s := &SpatialAimFile{Version: WeaponFireResidualVersion}
	width := ConditionalFireInputWidth
	for _, n := range []int{32, 16, 1} {
		l := DenseLayer{Weight: make([][]float64, n), Bias: make([]float64, n)}
		for i := range l.Weight {
			l.Weight[i] = make([]float64, width)
		}
		s.Layers = append(s.Layers, l)
		width = n
	}
	f := PPOFile{ConditionalFire: &SpatialAimFile{}, WeaponFireResidual: s}
	if err := validateWeaponFireResidual(f); err != nil {
		t.Fatal(err)
	}
	f.ConditionalFire = nil
	if validateWeaponFireResidual(f) == nil {
		t.Fatal("accepted missing legacy fire branch")
	}
	f.ConditionalFire = &SpatialAimFile{}
	s.Version = "unknown"
	if validateWeaponFireResidual(f) == nil {
		t.Fatal("accepted unknown weapon fire contract")
	}
	x := make([]float64, EquippedWeaponFeatureWidth)
	x[0], x[1123] = 1, 1
	if !weaponFireResidualActive(x) {
		t.Fatal("observed shotgun missing")
	}
	x[1123] = 0
	if weaponFireResidualActive(x) {
		t.Fatal("other weapon entered gate")
	}
	x[1123], x[0] = 1, 0
	if weaponFireResidualActive(x) || weaponFireResidualActive(nil) {
		t.Fatal("inactive player entered gate")
	}
}
