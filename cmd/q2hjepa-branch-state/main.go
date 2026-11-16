// Export only the first verified provider observation of diagnostic branches.
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
	"q2coopbot/internal/learningenv"
	"q2coopbot/internal/policy"
	"strings"
)

func hash(b []byte) string { h := sha256.Sum256(b); return hex.EncodeToString(h[:]) }
func run() error {
	root := flag.String("root", "", "completed diagnostic matrix")
	out := flag.String("out", "", "fresh first-state JSONL")
	flag.Parse()
	if *root == "" || *out == "" {
		return fmt.Errorf("root,out required")
	}
	abs, e := filepath.Abs(*root)
	if e != nil {
		return e
	}
	complete, e := os.ReadFile(filepath.Join(abs, "complete.json"))
	if e != nil {
		return e
	}
	var done struct {
		State      string `json:"state"`
		Diagnostic bool   `json:"diagnostic_weapon_branches"`
	}
	if e = json.Unmarshal(complete, &done); e != nil {
		return e
	}
	if done.State != "complete" || !done.Diagnostic {
		return fmt.Errorf("completed diagnostic matrix required")
	}
	var paths []string
	e = filepath.WalkDir(abs, func(p string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if !d.IsDir() && d.Name() == "steps.jsonl" {
			paths = append(paths, p)
		}
		return nil
	})
	if e != nil {
		return e
	}
	f, e := os.OpenFile(*out, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
	if e != nil {
		return e
	}
	defer f.Close()
	enc := json.NewEncoder(f)
	sources := map[string]string{}
	receipts := map[string]string{}
	count := 0
	for _, p := range paths {
		worker := filepath.Dir(filepath.Dir(p))
		caseDir := filepath.Dir(worker)
		reportPath := filepath.Join(caseDir, "report.json")
		rb, e := os.ReadFile(reportPath)
		if e != nil {
			return e
		}
		var r struct {
			Valid    bool `json:"provenance_valid"`
			Complete bool `json:"capture_complete"`
			Results  []struct {
				Seed    int    `json:"seed"`
				Root    string `json:"root"`
				Valid   bool   `json:"capture_valid"`
				Fixture struct {
					Split string `json:"split"`
				} `json:"generated_fixture"`
			} `json:"results"`
		}
		if e = json.Unmarshal(rb, &r); e != nil {
			return e
		}
		if !r.Valid || !r.Complete || len(r.Results) != 1 || !r.Results[0].Valid || !strings.EqualFold(filepath.Clean(worker), filepath.Clean(r.Results[0].Root)) {
			return fmt.Errorf("unverified single-case capture")
		}
		split := r.Results[0].Fixture.Split
		if split != "train" && split != "validation" {
			return fmt.Errorf("diagnostic preference source must be TRAIN/validation")
		}
		receipts[reportPath] = hash(rb)
		proofPath := filepath.Join(filepath.Dir(p), "report.json")
		rb, e = os.ReadFile(proofPath)
		if e != nil {
			return e
		}
		var proof struct {
			Proof struct {
				Accepted bool `json:"accepted"`
			} `json:"command_proof"`
			Sync  bool `json:"synchronous_confirmed"`
			Reset bool `json:"observed_reset_confirmed"`
		}
		if e = json.Unmarshal(rb, &proof); e != nil {
			return e
		}
		if !proof.Proof.Accepted || !proof.Sync || !proof.Reset {
			return fmt.Errorf("native proof missing")
		}
		receipts[proofPath] = hash(rb)
		b, e := os.ReadFile(p)
		if e != nil {
			return e
		}
		sources[p] = hash(b)
		scan := bufio.NewScanner(strings.NewReader(string(b)))
		scan.Buffer(make([]byte, 65536), 4*1024*1024)
		found := false
		for scan.Scan() {
			var s learningenv.Step
			if e = json.Unmarshal(scan.Bytes(), &s); e != nil {
				return e
			}
			o := s.Observation
			if s.Owner != "provider" || !strings.HasPrefix(s.Provider, "hjepa:") || s.Native == nil || s.Execution == nil || !s.Execution.Matched || !s.Execution.WindowExclusive || s.Native.EndFrame != s.Native.BeginFrame+1 || s.Native.Spawncount != o.Identity.Spawncount || s.Native.Actor != o.Identity.Actor || o.Health <= 0 {
				continue
			}
			x, e := policy.FeaturesForVersion(o, policy.WeaponFeatureVersion)
			if e != nil {
				return e
			}
			wx, e := policy.HJEPAWeaponFeatures(o)
			if e != nil {
				return e
			}
			mask, e := policy.WeaponAvailability(o)
			if e != nil {
				return e
			}
			if e = enc.Encode(map[string]any{"seed": r.Results[0].Seed, "split": split, "variant": filepath.Base(filepath.Dir(caseDir)), "identity": o.Identity, "features": x, "weapon_features": wx, "availability": mask, "weapon": o.Weapon, "source": p, "case_report": reportPath}); e != nil {
				return e
			}
			found = true
			count++
			break
		}
		if e = scan.Err(); e != nil {
			return e
		}
		if !found {
			return fmt.Errorf("no verified initial provider state: %s", p)
		}
	}
	if count == 0 {
		return fmt.Errorf("no branches")
	}
	if e = f.Sync(); e != nil {
		return e
	}
	b, e := json.MarshalIndent(map[string]any{"version": "hjepa_branch_states_v1", "rows": count, "sources_sha256": sources, "native_receipts_sha256": receipts, "complete_sha256": hash(complete), "scope": "One initial client-only provider state per diagnostic policy rollout; outcomes are not features."}, "", "  ")
	if e != nil {
		return e
	}
	mf, e := os.OpenFile(*out+".meta.json", os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
	if e != nil {
		return e
	}
	defer mf.Close()
	_, e = mf.Write(b)
	if e != nil {
		return e
	}
	fmt.Printf("Exported %d initial branch states\n", count)
	return nil
}
func main() {
	if e := run(); e != nil {
		fmt.Fprintln(os.Stderr, e)
		os.Exit(1)
	}
}
