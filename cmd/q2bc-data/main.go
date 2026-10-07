package main

import (
	"bufio"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"q2coopbot/internal/demodata"
	"q2coopbot/internal/policy"
)

type paths []string

func (p *paths) String() string     { return fmt.Sprint(*p) }
func (p *paths) Set(s string) error { *p = append(*p, s); return nil }

type row struct {
	Features    []float64          `json:"features"`
	Targets     []float64          `json:"targets"`
	Mask        []bool             `json:"mask"`
	Attack      float64            `json:"attack"`
	Vertical    int                `json:"vertical"`
	Observation policy.Observation `json:"observation"`
	Seed        int                `json:"seed"`
	Step        int                `json:"step"`
}

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
func run() error {
	var inputs paths
	flag.Var(&inputs, "dataset", "verified dataset directory (repeat)")
	out := flag.String("out", "", "fresh output directory")
	flag.Parse()
	if len(inputs) == 0 || *out == "" {
		return fmt.Errorf("datasets and fresh out required")
	}
	if err := os.Mkdir(*out, 0755); err != nil {
		return err
	}
	seen := map[int]string{}
	counts := map[string]int{}
	enc := map[string]*json.Encoder{}
	for _, s := range []string{"train", "validation", "test"} {
		f, e := os.Create(filepath.Join(*out, s+".jsonl"))
		if e != nil {
			return e
		}
		defer f.Close()
		enc[s] = json.NewEncoder(f)
	}
	for i, root := range inputs {
		data, e := os.ReadFile(filepath.Join(root, "report.json"))
		if e != nil {
			return e
		}
		var r demodata.Report
		if e = json.Unmarshal(data, &r); e != nil {
			return e
		}
		if !r.Ready {
			return fmt.Errorf("source not ready")
		}
		if demodata.IsQuerySelection(r.SelectionVersion) {
			return fmt.Errorf("counterfactual aim labels require q2bc-sequence-data; they are not applied teacher commands")
		}
		spec, e := os.ReadFile(filepath.Join(root, "spec.json"))
		if e != nil {
			return e
		}
		h := sha256.Sum256(spec)
		if hex.EncodeToString(h[:]) != r.SpecSHA {
			return fmt.Errorf("spec changed")
		}
		batches := []string{}
		unique := map[string]bool{}
		for _, s := range r.Sources {
			if _, ok := seen[s.Seed]; ok {
				return fmt.Errorf("reused seed %d", s.Seed)
			}
			seen[s.Seed] = s.Split
			if !unique[s.Batch] {
				batches = append(batches, s.Batch)
				unique[s.Batch] = true
			}
		}
		replay := filepath.Join(*out, fmt.Sprintf("source-%d", i))
		verified, e := demodata.Build(filepath.Join(root, "spec.json"), batches, replay)
		if e != nil {
			return e
		}
		for j, source := range r.Sources {
			if j >= len(verified.Sources) || source.Seed != verified.Sources[j].Seed {
				return fmt.Errorf("source assignments changed")
			}
			for key, value := range source.Files {
				if verified.Sources[j].Files[key] != value {
					return fmt.Errorf("source hash changed: %s", key)
				}
			}
		}
		for _, split := range []string{"train", "validation", "test"} {
			f, e := os.Open(filepath.Join(replay, split+".jsonl"))
			if e != nil {
				return e
			}
			scanner := bufio.NewScanner(f)
			scanner.Buffer(make([]byte, 65536), 2*1024*1024)
			for scanner.Scan() {
				var c demodata.Candidate
				if e = json.Unmarshal(scanner.Bytes(), &c); e != nil {
					f.Close()
					return e
				}
				if c.Source.Split != split || seen[c.Source.Seed] != split {
					f.Close()
					return fmt.Errorf("split leakage")
				}
				x, e := policy.Features(c.Observation)
				if e != nil {
					f.Close()
					return e
				}
				a := c.Target
				v := map[string]int{"release": 0, "jump": 1, "crouch": 2}[a.Vertical]
				attack := 0.0
				if a.Attack {
					attack = 1
				}
				sample := row{x, []float64{a.Forward, a.Side, a.YawDelta / 180, a.PitchDelta / 180}, []bool{c.Selection.Heads.Movement, c.Selection.Heads.Aim, c.Selection.Heads.Attack, c.Selection.Heads.Vertical}, attack, v, c.Observation, c.Source.Seed, c.Step}
				if e = enc[split].Encode(sample); e != nil {
					f.Close()
					return e
				}
				counts[split]++
			}
			e = scanner.Err()
			f.Close()
			if e != nil {
				return e
			}
		}
	}
	for _, split := range []string{"train", "validation", "test"} {
		if counts[split] == 0 {
			return fmt.Errorf("empty split")
		}
	}
	meta, _ := json.MarshalIndent(map[string]any{"version": "combat_bc_data_v1", "features": policy.FeatureVersion, "inputs": inputs, "counts": counts, "seed_splits": seen, "scope": "Revalidated masked teacher candidates; no tactical quality or unseen-condition acceptance"}, "", "  ")
	return os.WriteFile(filepath.Join(*out, "report.json"), meta, 0644)
}
