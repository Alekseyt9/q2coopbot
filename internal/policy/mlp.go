package policy

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
)

const MLPKind = "combat_bc_mlp_v1"
const MaxModelBytes = 16 * 1024 * 1024
const MaxHiddenWidth = 256

type DenseLayer struct {
	Weight [][]float64 `json:"weight"`
	Bias   []float64   `json:"bias"`
}
type MLPFile struct {
	Kind     string       `json:"kind"`
	Features string       `json:"feature_version"`
	Layers   []DenseLayer `json:"layers"`
}
type MLP struct {
	file    MLPFile
	version string
}

func LoadMLP(path string) (*MLP, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	if len(data) > MaxModelBytes {
		return nil, fmt.Errorf("model too large")
	}
	d := json.NewDecoder(bytes.NewReader(data))
	d.DisallowUnknownFields()
	var f MLPFile
	if err = d.Decode(&f); err != nil {
		return nil, err
	}
	if err = d.Decode(new(any)); err != io.EOF {
		return nil, fmt.Errorf("trailing model data")
	}
	if f.Kind != MLPKind || (f.Features != FeatureVersion && f.Features != AimFeatureVersion && f.Features != BBoxFeatureVersion && f.Features != TypedFeatureVersion) || len(f.Layers) != 3 {
		return nil, fmt.Errorf("unsupported BC architecture")
	}
	if err := validateFeatureLayers(f.Layers, 8, f.Features); err != nil {
		return nil, err
	}
	h := sha256.Sum256(data)
	return &MLP{f, "bc_mlp:" + hex.EncodeToString(h[:])}, nil
}
func validateLayers(layers []DenseLayer, outputs int) error {
	return validateFeatureLayers(layers, outputs, FeatureVersion)
}
func validateFeatureLayers(layers []DenseLayer, outputs int, version string) error {
	if len(layers) != 3 {
		return fmt.Errorf("three dense layers required")
	}
	x, err := FeaturesForVersion(Observation{}, version)
	if err != nil {
		return err
	}
	n := len(x)
	for i, l := range layers {
		if len(l.Weight) != len(l.Bias) || len(l.Bias) == 0 || len(l.Bias) > MaxHiddenWidth || i == 2 && len(l.Bias) != outputs {
			return fmt.Errorf("bad layer width")
		}
		for r, row := range l.Weight {
			if len(row) != n {
				return fmt.Errorf("bad layer input")
			}
			for _, v := range append(append([]float64{}, row...), l.Bias[r]) {
				if math.IsNaN(v) || math.IsInf(v, 0) || math.Abs(v) > 1e4 {
					return fmt.Errorf("bad weight")
				}
			}
		}
		n = len(l.Bias)
	}
	return nil
}
func (p *MLP) Version() string { return p.version }
func (p *MLP) Raw(o Observation) ([]float64, error) {
	version := p.file.Features
	if version == "" {
		version = FeatureVersion
	} // package-local legacy fixtures
	x, err := FeaturesForVersion(o, version)
	if err != nil {
		return nil, err
	}
	for i, l := range p.file.Layers {
		y := make([]float64, len(l.Bias))
		for r, row := range l.Weight {
			y[r] = l.Bias[r]
			for c, w := range row {
				y[r] += w * x[c]
			}
			if i < 2 {
				y[r] = math.Max(0, y[r])
			}
			if math.IsInf(y[r], 0) || math.IsNaN(y[r]) {
				return nil, fmt.Errorf("nonfinite inference")
			}
		}
		x = y
	}
	return x, nil
}
func (p *MLP) Decide(o Observation) (Action, error) {
	x, err := p.Raw(o)
	if err != nil {
		return Action{}, err
	}
	vertical := 0
	for i := 1; i < 3; i++ {
		if x[5+i] > x[5+vertical] {
			vertical = i
		}
	}
	a := Action{Version: ActionVersion, Identity: o.Identity, Forward: math.Tanh(x[0]), Side: math.Tanh(x[1]), YawDelta: 180 * math.Tanh(x[2]), PitchDelta: 180 * math.Tanh(x[3]), Attack: x[4] >= 0, Vertical: []string{"release", "jump", "crouch"}[vertical]}
	_, err = Command(o, a, [3]int16{})
	return a, err
}
