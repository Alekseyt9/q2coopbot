package policy

import "testing"

// Validate serialized dimensions only; no Go neural execution.
func TestSpatialCoarseContractDimensions(t *testing.T) {
	file := &SpatialAimFile{Version: SpatialCoarseAimVersion}
	width := SpatialAimInputWidth
	for _, rows := range []int{64, 32, 6} {
		layer := DenseLayer{Bias: make([]float64, rows), Weight: make([][]float64, rows)}
		for i := range layer.Weight {
			layer.Weight[i] = make([]float64, width)
		}
		file.Layers = append(file.Layers, layer)
		width = rows
	}
	if err := validateSpatialAim(file); err != nil {
		t.Fatal(err)
	}
	file.Version = SpatialAimVersion
	if validateSpatialAim(file) == nil {
		t.Fatal("v1 accepted v2 output layout")
	}
	file.Layers[2].Weight = file.Layers[2].Weight[:4]
	file.Layers[2].Bias = file.Layers[2].Bias[:4]
	if err := validateSpatialAim(file); err != nil {
		t.Fatal(err)
	}
	file.Version = SpatialCoarseAimVersion
	if validateSpatialAim(file) == nil {
		t.Fatal("v2 accepted incomplete coarse layout")
	}
}
