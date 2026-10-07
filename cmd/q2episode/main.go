package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"q2coopbot/internal/trainingepisodes"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
func run() error {
	registry := flag.String("registry", "scripts/scenarios/combat-training/index.json", "training episode registry")
	list := flag.Bool("list", false, "validate and list available/planned episodes")
	verify := flag.String("verify-plan", "", "recompute and verify a frozen generated execution plan")
	ids := flag.String("episodes", "", "comma-separated episode IDs")
	split := flag.String("split", "validation", "train, validation, test or confirmation")
	mode := flag.String("mode", "both", "rules, learned or both")
	model := flag.String("model", "", "frozen PPO weights for learned")
	count := flag.Int("count", 4, "episodes per descriptor, multiple of four")
	offset := flag.Int("seed-offset", 0, "offset inside the registered split")
	root := flag.String("root", ".", "repository root")
	out := flag.String("out", "", "fresh execution plan JSON")
	artifacts := flag.String("artifacts", "", "fresh runtime output root")
	flag.Parse()
	if *verify != "" {
		abs, err := filepath.Abs(*root)
		if err != nil {
			return err
		}
		return trainingepisodes.VerifyPlan(*verify, abs)
	}
	r, err := trainingepisodes.Load(*registry)
	if err != nil {
		return err
	}
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	if *list {
		return enc.Encode(r)
	}
	if *out == "" || *artifacts == "" {
		return fmt.Errorf("out and artifacts required")
	}
	abs, err := filepath.Abs(*root)
	if err != nil {
		return err
	}
	p, err := trainingepisodes.Build(r, abs, strings.Split(*ids, ","), *split, *mode, *model, *artifacts, *count, *offset)
	if err != nil {
		return err
	}
	f, err := os.OpenFile(*out, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0644)
	if err != nil {
		return err
	}
	defer f.Close()
	e := json.NewEncoder(f)
	e.SetIndent("", "  ")
	return e.Encode(p)
}
