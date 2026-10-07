// Compare the deployed Go policy to the offline nominal query labels.
package main

import (
	"bufio"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"q2coopbot/internal/aimquery"
	"q2coopbot/internal/demodata"
	"q2coopbot/internal/learningenv"
	"q2coopbot/internal/policy"
)

func fileHash(path string) (string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	h := sha256.Sum256(data)
	return hex.EncodeToString(h[:]), nil
}

type diagnostic struct {
	Source    demodata.Source    `json:"source"`
	Selection demodata.Selection `json:"selection"`
	Step      learningenv.Step   `json:"step"`
	Query     *aimquery.Label    `json:"counterfactual_aim_query"`
}
type metric struct {
	Context int     `json:"context"`
	Aim     int     `json:"aim_rows"`
	Sum     float64 `json:"-"`
	RMSE    float64 `json:"aim_rmse_degrees"`
}

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
func run() error {
	model := flag.String("model", "", "deterministic deployed weights")
	data := flag.String("data", "", "verified query sequence export")
	out := flag.String("out", "", "report JSON")
	flag.Parse()
	if *model == "" || *data == "" || *out == "" {
		return fmt.Errorf("model, data and out required")
	}
	var meta struct {
		Version  string            `json:"version"`
		Feature  string            `json:"feature_version"`
		Deferred bool              `json:"test_deferred"`
		SHA      map[string]string `json:"data_sha256"`
	}
	metadataPath := filepath.Join(*data, "report.json")
	blob, err := os.ReadFile(metadataPath)
	if err != nil {
		return err
	}
	if err = json.Unmarshal(blob, &meta); err != nil {
		return err
	}
	if meta.Version != "combat_bc_sequence_v1" || !meta.Deferred {
		return fmt.Errorf("verified deferred-test sequence export required")
	}
	receipts := map[string]string{}
	for _, path := range []string{*model, metadataPath, filepath.Join(*data, "train.jsonl"), filepath.Join(*data, "validation.jsonl")} {
		digest, err := fileHash(path)
		if err != nil {
			return err
		}
		receipts[path] = digest
	}
	for _, split := range []string{"train", "validation"} {
		if receipts[filepath.Join(*data, split+".jsonl")] != meta.SHA[split] {
			return fmt.Errorf("sequence data hash differs")
		}
	}
	// Never overwrite a prior review result.
	output, err := os.OpenFile(*out, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0644)
	if err != nil {
		return err
	}
	defer output.Close()
	folders, err := filepath.Glob(filepath.Join(*data, "source-*", "diagnostics.jsonl"))
	if err != nil {
		return err
	}
	if len(folders) == 0 {
		return fmt.Errorf("no verified query diagnostics")
	}
	metrics := map[string]*metric{"train": {}, "validation": {}}
	for _, path := range folders {
		digest, err := fileHash(path)
		if err != nil {
			return err
		}
		receipts[path] = digest
		file, err := os.Open(path)
		if err != nil {
			return err
		}
		scanner := bufio.NewScanner(file)
		scanner.Buffer(make([]byte, 65536), 8*1024*1024)
		var p *policy.PPO
		lastSeed := -1
		for scanner.Scan() {
			var d diagnostic
			if err = json.Unmarshal(scanner.Bytes(), &d); err != nil {
				file.Close()
				return err
			}
			s := d.Step
			if d.Selection.Version != demodata.AimQuerySelectionVersion {
				return fmt.Errorf("not a query corpus")
			}
			if s.Owner != "provider" || s.Observation.Identity.Life != 1 || s.Observation.Health <= 0 || s.Observation.AgeMS < 0 || s.Observation.AgeMS > 300 || s.Native == nil || s.Execution == nil || !s.Execution.Matched {
				continue
			}
			m := metrics[d.Source.Split]
			if m == nil {
				file.Close()
				return fmt.Errorf("unsupported split")
			}
			if p == nil || lastSeed != d.Source.Seed {
				p, err = policy.LoadPPO(*model)
				if err != nil {
					file.Close()
					return err
				}
				if p.IsStochastic() {
					file.Close()
					return fmt.Errorf("deterministic weights required")
				}
				if p.FeatureVersion() != meta.Feature {
					file.Close()
					return fmt.Errorf("feature contract differs")
				}
				lastSeed = d.Source.Seed
			}
			a, err := p.Decide(s.Observation)
			if err != nil {
				file.Close()
				return err
			}
			m.Context++
			if d.Selection.Quality == "rejected" || !d.Selection.Heads.Aim {
				continue
			}
			if d.Query == nil || d.Query.Action.Identity != s.Observation.Identity {
				file.Close()
				return fmt.Errorf("invalid query")
			}
			yaw := math.Remainder(a.YawDelta-d.Query.Action.YawDelta, 360)
			pitch := a.PitchDelta - d.Query.Action.PitchDelta
			m.Sum += (yaw*yaw + pitch*pitch) / 2
			m.Aim++
		}
		err = scanner.Err()
		file.Close()
		if err != nil {
			return err
		}
	}
	for _, m := range metrics {
		if m.Aim == 0 {
			return fmt.Errorf("empty aim split")
		}
		m.RMSE = math.Sqrt(m.Sum / float64(m.Aim))
	}
	for path, digest := range receipts {
		now, err := fileHash(path)
		if err != nil || now != digest {
			return fmt.Errorf("input changed during review")
		}
	}
	return json.NewEncoder(output).Encode(map[string]any{"version": "combat_go_query_review_v1", "metrics": metrics, "source_sha256": receipts, "scope": "Offline deterministic Go replay of verified provider context versus UNEXECUTED nominal query labels; not native hit accuracy or live acceptance."})
}
