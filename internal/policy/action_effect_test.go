package policy

import (
	"math"
	"reflect"
	"testing"
)

// Pure feature packing, geometry and metadata; no neural forward/score calls.
func TestActionEffectGeometryAndMetadata(t *testing.T) {
	c := make([]float64, 185)
	c[7], c[9], c[99], c[101], c[147], c[148], c[149], c[150] = 1, 1, 1, .5, 1, .25, -.375, .5
	x, err := MotorActionInputs(c, [2]float64{.1, 1})
	if err != nil || len(x) != 192 || x[190] != .1 || x[191] != 1 {
		t.Fatal(x, err)
	}
	c[162], c[164] = 1, .5
	x, err = MotorActionInputs(c, [2]float64{})
	if err != nil || math.Abs(x[188]) > 1e-9 || math.Abs(x[189]-1) > 1e-9 {
		t.Fatal("forward after yaw90", x, err)
	}
	c[162], c[164], c[163] = 0, 0, 1
	x, _ = MotorActionInputs(c, [2]float64{})
	if x[188] != 0 || x[189] != -1 {
		t.Fatal("Quake side sign", x)
	}
	g, err := ProjectMotorGeometry(c, make([]float64, 5))
	base, _ := ObservedActionGeometryForVersion(c, ActionGeometryCenteredVersion)
	if err != nil || !reflect.DeepEqual(g, base) {
		t.Fatal("zero forecast changed centered geometry", g, base, err)
	}
	g, err = ProjectMotorGeometry(c, []float64{.5, 0, 0, 0, 0})
	if err != nil || g[1] <= base[1] {
		t.Fatal("forward displacement did not bring target closer", g, err)
	}
	if _, err = ProjectMotorGeometry(c, []float64{0, 0, 0, 0, math.NaN()}); err == nil {
		t.Fatal("nonfinite forecast accepted")
	}
	p, d := motorColumns()
	m := &ActionEffectMotor{Version: "combat_action_effect_motor_v1", Hand: 2, InputWidth: 192, PositionColumns: p, DirectionColumns: d, Mean: make([]float64, 192), Std: make([]float64, 192), DisplacementScale: 32, AngleScale: 5}
	for i := range m.Std {
		m.Std[i] = 1
	}
	layers := func(width, hidden, out int) []DenseLayer {
		result := []DenseLayer{}
		for _, n := range []int{hidden, out} {
			l := DenseLayer{Weight: make([][]float64, n), Bias: make([]float64, n)}
			for i := range l.Weight {
				l.Weight[i] = make([]float64, width)
			}
			result = append(result, l)
			width = n
		}
		return result
	}
	m.Position = layers(76, 32, 3)
	m.Direction = layers(21, 16, 2)
	if err = validateActionEffectMotor(m); err != nil {
		t.Fatal(err)
	}
	m.Hand = 0
	if validateActionEffectMotor(m) == nil {
		t.Fatal("wrong hand accepted")
	}
	m.Hand = 2
	m.TargetSeconds = .1
	if validateActionEffectMotor(m) == nil {
		t.Fatal("unsupported target lead accepted")
	}
	m.TargetSeconds = 0
	m.DirectionColumns[0] = 99
	if validateActionEffectMotor(m) == nil {
		t.Fatal("enemy feature accepted in recoil head")
	}
}
