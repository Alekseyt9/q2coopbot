// Replay a frozen H-JEPA on native observations; no server outcomes are inputs.
package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"q2coopbot/internal/learningenv"
	"q2coopbot/internal/policy"
)

func run() error {
	model := flag.String("model", "", "frozen H-JEPA weights")
	root := flag.String("root", "", "native capture subtree")
	out := flag.String("out", "", "fresh summary JSON")
	require := flag.Bool("require-replay", false, "require captured proposals to match this model exactly")
	flag.Parse()
	p, err := policy.LoadHJEPA(*model)
	if err != nil {
		return err
	}
	frames, changed, proposed, accepted, mismatches := 0, 0, 0, 0, 0
	versions := map[string]int{}
	reasons := map[string]int{}
	err = filepath.WalkDir(*root, func(path string, d os.DirEntry, e error) error {
		if e != nil {
			return e
		}
		if d.IsDir() || d.Name() != "steps.jsonl" {
			return nil
		}
		b, e := os.ReadFile(filepath.Join(filepath.Dir(path), "report.json"))
		if e != nil {
			return e
		}
		var receipt struct {
			Proof struct {
				Accepted bool `json:"accepted"`
			} `json:"command_proof"`
		}
		if e = json.Unmarshal(b, &receipt); e != nil {
			return e
		}
		if !receipt.Proof.Accepted {
			return fmt.Errorf("unverified native data")
		}
		f, e := os.Open(path)
		if e != nil {
			return e
		}
		defer f.Close()
		scan := bufio.NewScanner(f)
		scan.Buffer(make([]byte, 65536), 4*1024*1024)
		for scan.Scan() {
			var s learningenv.Step
			if e = json.Unmarshal(scan.Bytes(), &s); e != nil {
				return e
			}
			if s.Owner != "provider" || s.Native == nil || s.Observation.Health <= 0 {
				continue
			}
			a, e := p.Decide(s.Observation)
			if e != nil {
				return e
			}
			audit := p.LastSearchAudit()
			motor, e := p.Proposal(s.Observation)
			if e != nil {
				return e
			}
			frames++
			versions[s.Provider]++
			if a != s.Action {
				mismatches++
			}
			if a != motor {
				changed++
			}
			if audit.ProposedChange {
				proposed++
			}
			if audit.Accepted {
				accepted++
			}
			for _, i := range s.Interventions {
				reasons[i.Reason]++
			}
		}
		return scan.Err()
	})
	if err != nil {
		return err
	}
	if frames == 0 {
		return fmt.Errorf("no native provider frames")
	}
	if *require && mismatches > 0 {
		return fmt.Errorf("%d replay mismatches", mismatches)
	}
	report := map[string]any{"version": "hjepa_search_audit_v1", "model": p.Version(), "native_frames": frames, "proposed_changes": proposed, "accepted_changes": accepted, "executed_model_changes": changed, "captured_proposal_mismatches": mismatches, "source_providers": versions, "native_guard_reasons": reasons, "scope": "Frozen replay of causal observations. Only --require-replay proves these proposals match the actual run; otherwise hypothetical diagnostic."}
	b, _ := json.MarshalIndent(report, "", "  ")
	f, err := os.OpenFile(*out, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
	if err != nil {
		return err
	}
	defer f.Close()
	if _, err = f.Write(b); err != nil {
		return err
	}
	fmt.Println(string(b))
	return nil
}
func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
