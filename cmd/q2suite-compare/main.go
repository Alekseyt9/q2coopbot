package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"q2coopbot/internal/harness/compare"
)

func run() error {
	p := flag.String("config", "", "comparison configuration JSON")
	flag.Parse()
	if *p == "" || flag.NArg() != 0 {
		return fmt.Errorf("usage: q2suite-compare --config comparison.json")
	}
	f, err := os.Open(*p)
	if err != nil {
		return err
	}
	defer f.Close()
	var cfg struct {
		Baseline  string `json:"baseline"`
		Candidate string `json:"candidate"`
		Output    string `json:"output"`
	}
	d := json.NewDecoder(f)
	d.DisallowUnknownFields()
	if err = d.Decode(&cfg); err != nil {
		return err
	}
	if d.Decode(new(any)) != io.EOF {
		return fmt.Errorf("trailing comparison configuration")
	}
	if cfg.Baseline == "" || cfg.Candidate == "" || cfg.Output == "" {
		return fmt.Errorf("baseline, candidate and output required")
	}
	return compareFiles(*p, cfg.Baseline, cfg.Candidate, cfg.Output)
}
func compareFiles(config, baseline, candidate, output string) error {
	resolve := func(p string) string {
		if filepath.IsAbs(p) {
			return p
		}
		return filepath.Join(filepath.Dir(config), p)
	}
	a, err := compare.Load(resolve(baseline))
	if err != nil {
		return err
	}
	b, err := compare.Load(resolve(candidate))
	if err != nil {
		return err
	}
	r := compare.Compare(a, b)
	data, err := json.MarshalIndent(r, "", "  ")
	if err != nil {
		return err
	}
	if err = os.WriteFile(resolve(output), data, 0600); err != nil {
		return err
	}
	fmt.Printf("compatible=%t groups=%d output=%s\n", r.Compatible, len(r.Groups), resolve(output))
	if !r.Compatible {
		return fmt.Errorf("suites are not fully comparable; inspect reasons in report")
	}
	return nil
}
func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
