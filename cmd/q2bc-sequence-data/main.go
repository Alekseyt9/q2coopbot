// Revalidate native teacher proof and retain masked chronological context for BC.
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
	"q2coopbot/internal/aimquery"
	"q2coopbot/internal/demodata"
	"q2coopbot/internal/learningenv"
	"q2coopbot/internal/policy"
)

type paths []string

func (p *paths) String() string     { return fmt.Sprint(*p) }
func (p *paths) Set(s string) error { *p = append(*p, s); return nil }
func hash(p string) (string, error) {
	b, e := os.ReadFile(p)
	if e != nil {
		return "", e
	}
	v := sha256.Sum256(b)
	return hex.EncodeToString(v[:]), nil
}
func read(p string, v any) error {
	b, e := os.ReadFile(p)
	if e != nil {
		return e
	}
	return json.Unmarshal(b, v)
}

type diagnostic struct {
	Source    demodata.Source    `json:"source"`
	Selection demodata.Selection `json:"selection"`
	Step      learningenv.Step   `json:"step"`
	Query     *aimquery.Label    `json:"counterfactual_aim_query,omitempty"`
}
type sample struct {
	LabelKind string          `json:"label_kind"`
	Features  []float64       `json:"features"`
	Targets   []float64       `json:"targets"`
	Mask      []bool          `json:"mask"`
	Attack    bool            `json:"attack"`
	Identity  policy.Identity `json:"identity"`
	Seed      int             `json:"seed"`
	Step      int             `json:"step"`
}

func convert(d diagnostic, feature string) (*sample, error) {
	s := d.Step
	if s.Observation.Identity.Life != 1 || s.Observation.Health <= 0 || s.Observation.AgeMS < 0 || s.Observation.AgeMS > 300 || s.Native == nil || s.Execution == nil || !s.Execution.Matched {
		return nil, nil
	}
	x, e := policy.FeaturesForVersion(s.Observation, feature)
	if e != nil {
		return nil, e
	}
	h := d.Selection.Heads
	mask := []bool{h.Movement, h.Aim, h.Attack, h.Vertical}
	if d.Selection.Quality == "rejected" {
		mask = []bool{false, false, false, false}
	}
	a := s.AppliedAction
	labelKind := "executed_teacher"
	if d.Selection.Quality == "rejected" {
		labelKind = "context"
	}
	if demodata.IsQuerySelection(d.Selection.Version) {
		if s.Owner != "provider" {
			return nil, nil
		}
		if mask[1] {
			coordinated := d.Selection.Version == demodata.CoordinatedQuerySelectionVersion
			version := aimquery.Version
			if coordinated {
				version = aimquery.CoordinatedVersion
			}
			if d.Query == nil || d.Query.Version != version || d.Query.Action.Identity != s.Observation.Identity || mask[0] != coordinated || mask[2] || mask[3] || h.Weapon {
				return nil, fmt.Errorf("invalid counterfactual aim mask/identity")
			}
			a = d.Query.Action
			labelKind = "counterfactual_nominal_aim"
			if coordinated {
				labelKind = "counterfactual_nominal_aim_world_input"
			}
			if _, err := policy.Command(s.Observation, a, [3]int16{}); err != nil {
				return nil, err
			}
		}
	}
	return &sample{LabelKind: labelKind, Features: x, Targets: []float64{a.Forward, a.Side, a.YawDelta / 180, a.PitchDelta / 180}, Mask: mask, Attack: a.Attack, Identity: s.Observation.Identity, Seed: d.Source.Seed, Step: s.Index}, nil
}
func main() {
	if e := run(); e != nil {
		fmt.Fprintln(os.Stderr, e)
		os.Exit(1)
	}
}
func run() error {
	var datasets paths
	flag.Var(&datasets, "dataset", "verified teacher dataset (repeat)")
	out := flag.String("out", "", "fresh output directory")
	feature := flag.String("features", policy.WeaponFeatureVersion, "feature contract")
	flag.Parse()
	if len(datasets) == 0 || *out == "" {
		return fmt.Errorf("datasets and fresh out required")
	}
	if e := os.Mkdir(*out, 0755); e != nil {
		return e
	}
	files := map[string]*os.File{}
	enc := map[string]*json.Encoder{}
	counts := map[string]int{}
	aims := map[string]int{}
	positive := map[string]int{}
	negative := map[string]int{}
	labelKinds := map[string]map[string]int{"train": {}, "validation": {}}
	seen := map[int]string{}
	receipts := map[string]string{}
	for _, split := range []string{"train", "validation"} {
		f, e := os.Create(filepath.Join(*out, split+".jsonl"))
		if e != nil {
			return e
		}
		defer f.Close()
		files[split] = f
		enc[split] = json.NewEncoder(f)
	}
	for i, root := range datasets {
		var original demodata.Report
		if e := read(filepath.Join(root, "report.json"), &original); e != nil {
			return e
		}
		if !original.Ready {
			return fmt.Errorf("teacher dataset not ready")
		}
		specPath := filepath.Join(root, "spec.json")
		digest, e := hash(specPath)
		if e != nil {
			return e
		}
		if digest != original.SpecSHA {
			return fmt.Errorf("teacher spec changed")
		}
		var spec demodata.Spec
		if e := read(specPath, &spec); e != nil {
			return e
		}
		if !spec.DeferredTest {
			return fmt.Errorf("sequence BC requires explicitly deferred final test")
		}
		batches := []string{}
		batchSet := map[string]bool{}
		for _, source := range original.Sources {
			if _, ok := seen[source.Seed]; ok {
				return fmt.Errorf("reused teacher seed")
			}
			seen[source.Seed] = source.Split
			if !batchSet[source.Batch] {
				batches = append(batches, source.Batch)
				batchSet[source.Batch] = true
			}
		}
		replay := filepath.Join(*out, fmt.Sprintf("source-%d", i))
		verified, e := demodata.Build(specPath, batches, replay)
		if e != nil {
			return e
		}
		if len(verified.Sources) != len(original.Sources) {
			return fmt.Errorf("source count changed")
		}
		for j, source := range original.Sources {
			actual := verified.Sources[j]
			if source.Seed != actual.Seed || source.Split != actual.Split {
				return fmt.Errorf("teacher assignment changed")
			}
			for key, value := range source.Files {
				if actual.Files[key] != value {
					return fmt.Errorf("teacher source changed: %s", key)
				}
			}
		}
		for _, source := range verified.Sources {
			for key, v := range source.Files {
				receipts[fmt.Sprintf("%d:%s", source.Seed, key)] = v
			}
		}
		receipts[specPath] = digest
		f, e := os.Open(filepath.Join(replay, "diagnostics.jsonl"))
		if e != nil {
			return e
		}
		scanner := bufio.NewScanner(f)
		scanner.Buffer(make([]byte, 65536), 4*1024*1024)
		last := map[int]policy.Identity{}
		for scanner.Scan() {
			var d diagnostic
			if e := json.Unmarshal(scanner.Bytes(), &d); e != nil {
				f.Close()
				return e
			}
			if seen[d.Source.Seed] != d.Source.Split || enc[d.Source.Split] == nil {
				f.Close()
				return fmt.Errorf("split leakage")
			}
			row, e := convert(d, *feature)
			if e != nil {
				f.Close()
				return e
			}
			if row == nil {
				continue
			}
			if prev, ok := last[row.Seed]; ok && policy.SameLife(prev, row.Identity) && prev.Frame >= row.Identity.Frame {
				f.Close()
				return fmt.Errorf("duplicate or reordered context frame")
			}
			last[row.Seed] = row.Identity
			if e := enc[d.Source.Split].Encode(row); e != nil {
				f.Close()
				return e
			}
			counts[d.Source.Split]++
			labelKinds[d.Source.Split][row.LabelKind]++
			if row.Mask[1] {
				aims[d.Source.Split]++
			}
			if row.Mask[2] {
				if row.Attack {
					positive[d.Source.Split]++
				} else {
					negative[d.Source.Split]++
				}
			}
		}
		e = scanner.Err()
		f.Close()
		if e != nil {
			return e
		}
	}
	dataHashes := map[string]string{}
	for split, f := range files {
		if counts[split] == 0 || aims[split] == 0 {
			return fmt.Errorf("empty sequence/aim split")
		}
		if e := f.Sync(); e != nil {
			return e
		}
		if e := f.Close(); e != nil {
			return e
		}
		v, e := hash(filepath.Join(*out, split+".jsonl"))
		if e != nil {
			return e
		}
		dataHashes[split] = v
	}
	meta := map[string]any{"version": "combat_bc_sequence_v1", "feature_version": *feature, "counts": counts, "aim_rows": aims, "attack_positive": positive, "attack_negative": negative, "data_sha256": dataHashes, "source_sha256": receipts, "seed_splits": seen, "test_deferred": true, "scope": "Native-verified first-life teacher context, rejected labels masked; reset at identity/frame gaps. Final test not collected. Tracking conventions are not shot credit or tactical acceptance."}
	meta["label_kinds"] = labelKinds
	meta["scope"] = "Native-verified first-life context; executed teacher labels and separately marked UNEXECUTED nominal aim queries. Rejected labels masked; reset at identity/frame gaps. Final test deferred. No query hit/optimal-target/tactical proof or query rewards."
	if executable, err := os.Executable(); err == nil {
		digest, err := hash(executable)
		if err != nil {
			return err
		}
		meta["exporter_sha256"] = digest
	}
	b, e := json.MarshalIndent(meta, "", "  ")
	if e != nil {
		return e
	}
	return os.WriteFile(filepath.Join(*out, "report.json"), b, 0644)
}
