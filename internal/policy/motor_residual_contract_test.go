package policy

import (
	"math"
	"testing"
)

// Contract/layout only: this test does not execute a neural network.
func TestMotorResidualContractAndInputs(t *testing.T) {
	f := PPOFile{Features: EquippedWeaponFeatureVersion, AimModeHead: PrecisionHeadVersion,
		SpatialAim: &SpatialAimFile{}, SampledMotionAim: &SpatialAimFile{},
		MotorResidual: &SpatialAimFile{Version: MotorResidualVersion}}
	width := MotorResidualInputWidth
	for _, out := range []int{64, 32, 2} {
		layer := DenseLayer{Bias: make([]float64, out), Weight: make([][]float64, out)}
		for i := range layer.Weight {
			layer.Weight[i] = make([]float64, width)
		}
		f.MotorResidual.Layers = append(f.MotorResidual.Layers, layer)
		width = out
	}
	if err := validateMotorResidual(f); err != nil {
		t.Fatal(err)
	}
	features := make([]float64, 1147)
	for i := range features {
		features[i] = float64(i)
	}
	x := motorResidualInputs(features)
	if len(x) != 756 || x[168] != 168 || x[169] != 466 || x[488] != 785 || x[489] != 854 || x[755] != 1120 {
		t.Fatal("motor input offsets", len(x))
	}
	f.MotorResidual.Layers[0].Weight[0][0] = math.NaN()
	if validateMotorResidual(f) == nil {
		t.Fatal("accepted nonfinite weight")
	}
	f.MotorResidual.Layers[0].Weight[0][0] = 0
	f.SampledMotionAim = nil
	if validateMotorResidual(f) == nil {
		t.Fatal("accepted missing conditional controller")
	}
	if err := validateMotorResidual(PPOFile{}); err != nil {
		t.Fatal(err)
	}
}
