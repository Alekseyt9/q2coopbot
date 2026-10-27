package policy

import "testing"

// Input packing/metadata only. Numerical neural inference is verified on CUDA.
func TestConditionalFireContextContract(t *testing.T) {
	f := make([]float64, 1147)
	f[WeaponMaskOffset] = 1
	for i := 169; i < 218; i++ {
		f[i] = float64(i) / 1000
	}
	for i := 259; i < 296; i++ {
		f[i] = float64(i) / 1000
	}
	legacy, err := conditionalFireInputsForVersion(f, Sample{}, ConditionalFireVersion)
	if err != nil {
		t.Fatal(err)
	}
	current, err := conditionalFireInputsForVersion(f, Sample{}, ConditionalFireContextVersion)
	if err != nil {
		t.Fatal(err)
	}
	if len(current) != 267 {
		t.Fatal("missing observed context")
	}
	for i, v := range legacy {
		if current[i] != v {
			t.Fatal("legacy prefix changed")
		}
	}
	for i, v := range f[169:218] {
		if current[181+i] != v {
			t.Fatal("projectile packing")
		}
	}
	for i, v := range f[259:296] {
		if current[230+i] != v {
			t.Fatal("prop packing")
		}
	}
	if _, err := conditionalFireInputsForVersion(f, Sample{}, "unknown"); err == nil {
		t.Fatal("unknown version accepted")
	}
	spec := &SpatialAimFile{Version: ConditionalFireContextVersion}
	width := ConditionalFireContextInputWidth
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
	spec.Version = ConditionalFireVersion
	if validateConditionalFireSpec(spec) == nil {
		t.Fatal("new width accepted under old contract")
	}
}
