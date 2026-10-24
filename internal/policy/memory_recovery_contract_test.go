package policy

import (
	"math"
	"testing"
)

// Header/input checks only: no neural forward or numerical model comparison on Go.
func TestMemoryRecoveryContractAndInputs(t *testing.T) {
	spec := &SpatialAimFile{Version: MemoryRecoveryVersion}
	width := MemoryRecoveryInputWidth
	for _, n := range []int{32, 16, 3} {
		l := DenseLayer{Weight: make([][]float64, n), Bias: make([]float64, n)}
		for i := range l.Weight {
			l.Weight[i] = make([]float64, width)
		}
		spec.Layers = append(spec.Layers, l)
		width = n
	}
	f := PPOFile{Features: EquippedWeaponFeatureVersion, AimModeHead: PrecisionHeadVersion, SpatialAim: &SpatialAimFile{}, MemoryRecovery: spec}
	if err := validateMemoryRecovery(f); err != nil {
		t.Fatal(err)
	}
	spec.Layers[2].Bias[0] = math.NaN()
	if validateMemoryRecovery(f) == nil {
		t.Fatal("accepted nonfinite parameter")
	}
	spec.Layers[2].Bias[0] = 0
	spec.Version = "unsupported"
	if validateMemoryRecovery(f) == nil {
		t.Fatal("accepted unsupported version")
	}
	features, raw := make([]float64, EquippedWeaponFeatureWidth), make([]float64, PrecisionOutputWidth)
	features[881], features[882], features[1125] = 1, .05, 1
	raw[2], raw[3], raw[4] = 31, -32, .75
	x, active, err := memoryRecoveryInputs(features, raw)
	if err != nil || !active || len(x) != 46 || x[43] != 20 || x[44] != -20 || x[45] != .75 {
		t.Fatalf("input contract: active=%v err=%v", active, err)
	}
	features[882] = .055
	if _, active, _ := memoryRecoveryInputs(features, raw); active {
		t.Fatal("stale memory activated projection")
	}
	features[882], features[73] = .005, 1
	if _, active, _ := memoryRecoveryInputs(features, raw); active {
		t.Fatal("visible enemy activated projection")
	}
	if _, _, err := memoryRecoveryInputs(features[:12], raw); err == nil {
		t.Fatal("accepted wrong input width")
	}
	raw[4] = math.Inf(1)
	if _, _, err := memoryRecoveryInputs(features, raw); err == nil {
		t.Fatal("accepted infinite raw value")
	}
}
