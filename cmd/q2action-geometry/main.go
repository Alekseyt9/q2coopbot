// Pure observation/action packing and geometry export; never evaluates a neural model.
package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"q2coopbot/internal/policy"
)

func main() {
	input := flag.String("input", "", "input JSONL")
	out := flag.String("out", "", "fresh output JSONL")
	version := flag.String("geometry-version", policy.ActionGeometryVersion, "versioned geometry contract")
	motor := flag.Bool("motor-inputs", false, "also export pure observed/action motor inputs, never NN outputs")
	flag.Parse()
	if err := run(*input, *out, *version, *motor); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run(input, out, version string, exportMotor bool) error {
	if input == "" || out == "" {
		return fmt.Errorf("input and out required")
	}
	f, err := os.Open(input)
	if err != nil {
		return err
	}
	defer f.Close()
	g, err := os.OpenFile(out, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
	if err != nil {
		return err
	}
	defer g.Close()
	w := bufio.NewWriter(g)
	defer w.Flush()
	encoder := json.NewEncoder(w)
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 65536), 8*1024*1024)
	for scanner.Scan() {
		var row struct {
			Timing      [2]float64          `json:"timing"`
			Forecast    []float64           `json:"forecast"`
			Causal      []float64           `json:"causal"`
			Observation *policy.Observation `json:"observation"`
			Sample      *policy.Sample      `json:"sample"`
		}
		if err := json.Unmarshal(scanner.Bytes(), &row); err != nil {
			return err
		}
		if row.Causal == nil {
			if row.Observation == nil || row.Sample == nil {
				return fmt.Errorf("missing observation/sample")
			}
			features, err := policy.FeaturesForVersion(*row.Observation, policy.EquippedWeaponFeatureVersion)
			if err != nil {
				return err
			}
			row.Causal, err = policy.GeometryActionInputs(features, *row.Sample)
			if err != nil {
				return err
			}
			row.Timing = [2]float64{float64(row.Observation.AgeMS) / 300, float64(row.Observation.PreviousCommand.Msec) / 100}
		}
		var geometry []float64
		if version == policy.ActionEffectGeometryVersion {
			geometry, err = policy.ProjectMotorGeometry(row.Causal, row.Forecast)
		} else {
			geometry, err = policy.ObservedActionGeometryForVersion(row.Causal, version)
		}
		if err != nil {
			return err
		}
		var motorInput []float64
		if exportMotor {
			motorInput, err = policy.MotorActionInputs(row.Causal, row.Timing)
			if err != nil {
				return err
			}
		}
		if err := encoder.Encode(struct {
			Causal   []float64 `json:"causal"`
			Geometry []float64 `json:"geometry"`
			Motor    []float64 `json:"motor_inputs,omitempty"`
		}{row.Causal, geometry, motorInput}); err != nil {
			return err
		}
	}
	return scanner.Err()
}
