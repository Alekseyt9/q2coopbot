// Independent native-Go parity check for CUDA-exported H-JEPA networks.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"q2coopbot/internal/policy"
)

func run() error {
	model := flag.String("model", "", "H-JEPA weights")
	fixture := flag.String("parity", "", "CUDA reference fixture")
	motorOnly := flag.Bool("motor-only", false, "check independent distilled motor fixture")
	flag.Parse()
	p, e := policy.LoadHJEPA(*model)
	if e != nil {
		return e
	}
	b, e := os.ReadFile(*fixture)
	if e != nil {
		return e
	}
	if *motorOnly {
		var f struct {
			Features [][]float64 `json:"features"`
			Outputs  [][]float64 `json:"outputs"`
		}
		if e = json.Unmarshal(b, &f); e != nil {
			return e
		}
		if len(f.Features) == 0 || len(f.Features) != len(f.Outputs) {
			return fmt.Errorf("invalid motor fixture")
		}
		maxError := 0.
		for i, x := range f.Features {
			_, raw, _, err := p.Replay(x, []float64{0, 0, 0, 0, 0, 1, 0, 0})
			if err != nil {
				return err
			}
			if len(f.Outputs[i]) != len(raw) {
				return fmt.Errorf("motor parity shape")
			}
			for j, v := range raw {
				d := math.Abs(v - f.Outputs[i][j])
				if math.IsNaN(d) || math.IsInf(d, 0) || d > 1e-8 {
					return fmt.Errorf("motor parity mismatch")
				}
				maxError = math.Max(maxError, d)
			}
		}
		return json.NewEncoder(os.Stdout).Encode(map[string]any{"accepted": true, "motor_only": true, "max_abs_error": maxError, "values_compared": len(f.Features) * 8})
	}
	var f struct {
		Features    [][]float64   `json:"features"`
		Latents     [][][]float64 `json:"latents"`
		Actor       [][]float64   `json:"actor"`
		Actions     [][]float64   `json:"actions"`
		Predictions [][][]float64 `json:"predictions"`
	}
	if e = json.Unmarshal(b, &f); e != nil {
		return e
	}
	n := len(f.Features)
	if n == 0 || len(f.Actions) != n || len(f.Actor) != n || len(f.Latents) != 3 || len(f.Predictions) != 3 {
		return fmt.Errorf("invalid parity fixture")
	}
	maxError := 0.
	compared := 0
	check := func(a, b []float64) error {
		if len(a) != len(b) {
			return fmt.Errorf("parity shape mismatch")
		}
		for i, v := range a {
			d := math.Abs(v - b[i])
			if math.IsNaN(d) || math.IsInf(d, 0) || d > 1e-8 {
				return fmt.Errorf("parity mismatch %.12g", d)
			}
			maxError = math.Max(maxError, d)
			compared++
		}
		return nil
	}
	for i, x := range f.Features {
		z, actor, pred, e := p.Replay(x, f.Actions[i])
		if e != nil {
			return e
		}
		if e = check(actor, f.Actor[i]); e != nil {
			return e
		}
		for l := 0; l < 3; l++ {
			if len(f.Latents[l]) != n || len(f.Predictions[l]) != n {
				return fmt.Errorf("parity batch shape")
			}
			if e = check(z[l], f.Latents[l][i]); e != nil {
				return e
			}
			if e = check(pred[l], f.Predictions[l][i]); e != nil {
				return e
			}
		}
	}
	return json.NewEncoder(os.Stdout).Encode(map[string]any{"accepted": true, "values_compared": compared, "max_abs_error": maxError, "provider": p.Version()})
}
func main() {
	if e := run(); e != nil {
		fmt.Fprintln(os.Stderr, e)
		os.Exit(1)
	}
}
