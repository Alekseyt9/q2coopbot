package policy

import (
	"fmt"
	"math"
	"reflect"
)

const ActionEffectGeometryVersion = "combat_predicted_action_geometry_v1"

type ActionEffectMotor struct {
	Version           string       `json:"version"`
	Hand              int          `json:"hand"`
	InputWidth        int          `json:"input_width"`
	PositionColumns   []int        `json:"position_columns"`
	DirectionColumns  []int        `json:"direction_columns"`
	Mean              []float64    `json:"mean"`
	Std               []float64    `json:"std"`
	Position          []DenseLayer `json:"position_layers"`
	Direction         []DenseLayer `json:"direction_layers"`
	TargetSeconds     float64      `json:"target_extrapolation_seconds"`
	DisplacementScale float64      `json:"displacement_scale"`
	AngleScale        float64      `json:"angle_scale_degrees"`
}

func motorColumns() ([]int, []int) {
	var p, d []int
	add := func(to *[]int, from, end int) {
		for i := from; i < end; i++ {
			*to = append(*to, i)
		}
	}
	add(&p, 0, 6)
	add(&p, 13, 73)
	add(&p, 185, 190)
	add(&p, 166, 169)
	add(&p, 190, 192)
	add(&d, 0, 6)
	add(&d, 8, 10)
	add(&d, 15, 17)
	add(&d, 181, 188)
	d = append(d, 165, 161)
	add(&d, 190, 192)
	return p, d
}

func validateActionEffectMotor(m *ActionEffectMotor) error {
	p, d := motorColumns()
	if m == nil || m.Version != "combat_action_effect_motor_v1" || m.Hand != 2 || m.InputWidth != 192 ||
		m.TargetSeconds != 0 || m.DisplacementScale != 32 || m.AngleScale != 5 ||
		!reflect.DeepEqual(m.PositionColumns, p) || !reflect.DeepEqual(m.DirectionColumns, d) || len(m.Mean) != 192 || len(m.Std) != 192 {
		return fmt.Errorf("invalid action-effect motor contract")
	}
	for _, values := range [][]float64{m.Mean, m.Std} {
		for _, v := range values {
			if math.IsNaN(v) || math.IsInf(v, 0) || math.Abs(v) > 1e6 {
				return fmt.Errorf("invalid motor normalization")
			}
		}
	}
	for _, v := range m.Std {
		if v <= 0 {
			return fmt.Errorf("invalid motor std")
		}
	}
	for _, b := range []struct {
		layers             []DenseLayer
		width, hidden, out int
	}{{m.Position, 76, 32, 3}, {m.Direction, 21, 16, 2}} {
		if len(b.layers) != 2 {
			return fmt.Errorf("invalid motor layers")
		}
		width := b.width
		for i, output := range []int{b.hidden, b.out} {
			l := b.layers[i]
			if len(l.Weight) != output || len(l.Bias) != output {
				return fmt.Errorf("invalid motor outputs")
			}
			for r, row := range l.Weight {
				if len(row) != width {
					return fmt.Errorf("invalid motor inputs")
				}
				for _, v := range append(append([]float64(nil), row...), l.Bias[r]) {
					if math.IsNaN(v) || math.IsInf(v, 0) || math.Abs(v) > 1e6 {
						return fmt.Errorf("invalid motor parameter")
					}
				}
			}
			width = output
		}
	}
	return nil
}

// MotorActionInputs is pure observed/action geometry; no learned scores.
func MotorActionInputs(c []float64, timing [2]float64) ([]float64, error) {
	if len(c) != 185 {
		return nil, fmt.Errorf("motor inputs require causal185")
	}
	for _, v := range append(append([]float64(nil), c...), timing[:]...) {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return nil, fmt.Errorf("nonfinite motor input")
		}
	}
	yaw := math.Atan2(c[6], c[7])
	cy, sy := math.Cos(yaw), math.Sin(yaw)
	delta := c[164] * math.Pi
	cd, sd := math.Cos(delta), math.Sin(delta)
	x := append([]float64(nil), c...)
	x = append(x, cy*c[10]+sy*c[11], -sy*c[10]+cy*c[11], c[12], c[162]*cd+c[163]*sd, c[162]*sd-c[163]*cd, timing[0], timing[1])
	return x, nil
}

// ProjectMotorGeometry accepts an already predicted displacement/angle. It is
// pure geometry, reusable by offline packing without evaluating a neural model.
func ProjectMotorGeometry(c, forecast []float64) ([]float64, error) {
	if len(c) != 185 || len(forecast) != 5 {
		return nil, fmt.Errorf("invalid projected geometry width")
	}
	x := append([]float64(nil), c...)
	for i, v := range forecast {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return nil, fmt.Errorf("nonfinite motor forecast")
		}
		if i < 3 {
			x[101+i] -= v * 32 / 512
		}
	}
	x[164] += forecast[4] * 5 / 180
	x[165] += forecast[3] * 5 / 180
	return ObservedActionGeometryForVersion(x, ActionGeometryCenteredVersion)
}

// forecastMotor is evaluated only by live aim-ranking Decisions. Offline Go
// exporters/tests use MotorActionInputs/ProjectMotorGeometry without this NN.
func forecastMotor(m *ActionEffectMotor, c []float64, timing [2]float64) ([]float64, error) {
	x, err := MotorActionInputs(c, timing)
	if err != nil {
		return nil, err
	}
	for i := range x {
		x[i] = (x[i] - m.Mean[i]) / m.Std[i]
	}
	var result []float64
	for _, b := range []struct {
		columns []int
		layers  []DenseLayer
	}{{m.PositionColumns, m.Position}, {m.DirectionColumns, m.Direction}} {
		v := make([]float64, len(b.columns))
		for i, column := range b.columns {
			v[i] = x[column]
		}
		for i, layer := range b.layers {
			y := make([]float64, len(layer.Bias))
			for r, row := range layer.Weight {
				y[r] = layer.Bias[r]
				for k, w := range row {
					y[r] += w * v[k]
				}
				if i == 0 {
					y[r] = math.Max(0, y[r])
				}
				if math.IsNaN(y[r]) || math.IsInf(y[r], 0) {
					return nil, fmt.Errorf("nonfinite motor output")
				}
			}
			v = y
		}
		result = append(result, v...)
	}
	return result, nil
}
