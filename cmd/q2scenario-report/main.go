package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"q2coopbot/internal/harness"
)

func run() error {
	path := flag.String("config", "", "report config JSON")
	flag.Parse()
	if *path == "" || flag.NArg() != 0 {
		return fmt.Errorf("usage: q2scenario-report --config report.json")
	}
	var cfg struct {
		Scenario string `json:"scenario"`
		Actor    string `json:"actor_trace"`
		Bot      string `json:"bot_trace"`
		Output   string `json:"output"`
	}
	f, err := os.Open(*path)
	if err != nil {
		return err
	}
	defer f.Close()
	d := json.NewDecoder(f)
	d.DisallowUnknownFields()
	if err = d.Decode(&cfg); err != nil {
		return err
	}
	if d.Decode(new(any)) != io.EOF {
		return fmt.Errorf("trailing report config")
	}
	resolve := func(p string) string {
		if filepath.IsAbs(p) {
			return p
		}
		return filepath.Join(filepath.Dir(*path), p)
	}
	if cfg.Scenario == "" || cfg.Actor == "" || cfg.Bot == "" || cfg.Output == "" {
		return fmt.Errorf("report config requires all paths")
	}
	s, err := harness.Load(resolve(cfg.Scenario))
	if err != nil {
		return err
	}
	a, err := harness.ReadTrace(resolve(cfg.Actor))
	if err != nil {
		return err
	}
	b, err := harness.ReadTrace(resolve(cfg.Bot))
	if err != nil {
		return err
	}
	r := harness.Analyze(s, a, b)
	data, err := json.MarshalIndent(r, "", "  ")
	if err != nil {
		return err
	}
	if err = os.WriteFile(resolve(cfg.Output), data, 0600); err != nil {
		return err
	}
	fmt.Println(string(data))
	if r.State != "passed" {
		return fmt.Errorf("scenario %s: %s", r.State, r.Reason)
	}
	return nil
}
func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
