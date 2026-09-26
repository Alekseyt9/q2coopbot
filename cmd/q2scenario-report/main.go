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
		ServerLog string `json:"server_log"`
		Session   string `json:"session"`
		Scenario  string `json:"scenario"`
		Actor     string `json:"actor_trace"`
		Bot       string `json:"bot_trace"`
		Output    string `json:"output"`
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
	if (cfg.Scenario == "") == (cfg.Session == "") || cfg.Actor == "" || cfg.Bot == "" || cfg.Output == "" {
		return fmt.Errorf("report config requires all paths")
	}
	a, err := harness.ReadTrace(resolve(cfg.Actor))
	if err != nil {
		return err
	}
	b, err := harness.ReadTrace(resolve(cfg.Bot))
	if err != nil {
		return err
	}
	if cfg.Session != "" {
		if cfg.ServerLog == "" {
			return fmt.Errorf("session report requires server_log for applied command verification")
		}
		s, err := harness.LoadSession(resolve(cfg.Session))
		if err != nil {
			return err
		}
		r := harness.AnalyzeSession(s, a, b)
		applied, err := harness.ReadAppliedCommands(resolve(cfg.ServerLog))
		if err != nil {
			r.Accepted = false
			r.State = "trace_invalid"
			r.Reason = err.Error()
		} else {
			proof := harness.VerifyAppliedCommands(b, applied)
			r.Commands = &proof
			if !proof.Accepted {
				r.Accepted = false
				r.State = "trace_invalid"
				r.Reason = "observer commands: " + proof.Reason
			}
		}
		return writeReport(resolve(cfg.Output), r, r.Accepted)
	}
	s, err := harness.Load(resolve(cfg.Scenario))
	if err != nil {
		return err
	}
	r := harness.Analyze(s, a, b)
	return writeReport(resolve(cfg.Output), r, r.Accepted)
}

func writeReport(output string, r any, accepted bool) error {
	data, err := json.MarshalIndent(r, "", "  ")
	if err != nil {
		return err
	}
	if err = os.WriteFile(output, data, 0600); err != nil {
		return err
	}
	fmt.Println(string(data))
	if !accepted {
		return fmt.Errorf("report rejected; see %s", output)
	}
	return nil
}
func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
