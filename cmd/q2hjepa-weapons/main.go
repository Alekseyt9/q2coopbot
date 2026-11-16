// Export frozen U1 weapon probabilities on verified causal client observations.
// These are hypothetical distillation targets, never executed-action labels.
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

func digest(b []byte) string { h := sha256.Sum256(b); return hex.EncodeToString(h[:]) }
func run() error {
	root := flag.String("root", "", "verified native capture root")
	model := flag.String("model", "", "frozen source PPO")
	out := flag.String("out", "", "fresh output JSONL")
	split := flag.String("split", "", "train or validation, enforced against native fixture")
	flag.Parse()
	if *root == "" || *model == "" || *out == "" || (*split != "train" && *split != "validation") {
		return fmt.Errorf("root,model,out and train/validation split required")
	}
	absRoot, e := filepath.Abs(*root)
	if e != nil {
		return e
	}
	*root = absRoot
	mb, e := os.ReadFile(*model)
	if e != nil {
		return e
	}
	var paths []string
	e = filepath.WalkDir(*root, func(p string, d os.DirEntry, err error) error {
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
	seeds := map[int]bool{}
	hist := map[string]int{}
	count := 0
	for _, path := range paths {
		p, e := policy.LoadPPO(*model)
		if e != nil {
			return e
		}
		if p.WeaponHeadVersion() != policy.WeaponHeadVersion {
			return fmt.Errorf("teacher has no masked weapon head")
		}
		worker := filepath.Dir(filepath.Dir(path))
		caseReport := filepath.Join(filepath.Dir(worker), "report.json")
		rb, e := os.ReadFile(caseReport)
		if e != nil {
			return e
		}
		var cr struct {
			Valid    bool `json:"provenance_valid"`
			Complete bool `json:"capture_complete"`
			Results  []struct {
				Root    string `json:"root"`
				Valid   bool   `json:"capture_valid"`
				Seed    int    `json:"seed"`
				Fixture struct {
					Split string `json:"split"`
				} `json:"generated_fixture"`
			} `json:"results"`
		}
		if e = json.Unmarshal(rb, &cr); e != nil {
			return e
		}
		if !cr.Valid || !cr.Complete {
			return fmt.Errorf("unverified native case")
		}
		found := false
		seed := 0
		for _, r := range cr.Results {
			if strings.EqualFold(filepath.Clean(r.Root), filepath.Clean(worker)) {
				if !r.Valid || r.Fixture.Split != *split {
					return fmt.Errorf("invalid capture or split: %s", path)
				}
				found = true
				seed = r.Seed
			}
		}
		if !found {
			return fmt.Errorf("native worker binding missing")
		}
		seeds[seed] = true
		receipts[caseReport] = digest(rb)
		rb, e = os.ReadFile(filepath.Join(filepath.Dir(path), "report.json"))
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
			return fmt.Errorf("native execution proof missing")
		}
		receipts[filepath.Join(filepath.Dir(path), "report.json")] = digest(rb)
		b, e := os.ReadFile(path)
		if e != nil {
			return e
		}
		sources[path] = digest(b)
		scan := bufio.NewScanner(strings.NewReader(string(b)))
		scan.Buffer(make([]byte, 65536), 4*1024*1024)
		for scan.Scan() {
			var s learningenv.Step
			if e = json.Unmarshal(scan.Bytes(), &s); e != nil {
				return e
			}
			o := s.Observation
			if o.Health <= 0 || o.Version != policy.ObservationVersion || o.AgeMS < 0 || o.AgeMS > 300 {
				continue
			}
			if _, e = p.Decide(o); e != nil {
				return e
			}
			// Provider ownership proves the setup barrier is released. Native
			// snapshot frame98 can correspond to game release100: never guess it.
			if s.Owner != "provider" || !strings.HasPrefix(s.Provider, "ppo:") || s.Native == nil || s.Native.EndFrame != s.Native.BeginFrame+1 || s.Native.Spawncount != o.Identity.Spawncount || s.Native.Actor != o.Identity.Actor || s.Execution == nil || !s.Execution.Matched || !s.Execution.WindowExclusive {
				continue
			}
			x, e := policy.FeaturesForVersion(o, policy.WeaponFeatureVersion)
			if e != nil {
				return e
			}
			y := p.LastWeaponDistribution()
			mask, e := policy.WeaponAvailability(o)
			if e != nil {
				return e
			}
			if len(y) != len(mask) {
				return fmt.Errorf("teacher distribution missing")
			}
			best := 0
			for i, v := range y {
				if !mask[i] && v != 0 {
					return fmt.Errorf("unavailable teacher label")
				}
				if v > y[best] {
					best = i
				}
			}
			hist[policy.WeaponChoices()[best]]++
			row := map[string]any{"seed": seed, "split": *split, "identity": o.Identity, "features": x, "teacher_weapon_probabilities": y, "label_kind": "hypothetical_u1_weapon_distribution_v1"}
			if e = enc.Encode(row); e != nil {
				return e
			}
			count++
		}
		if e = scan.Err(); e != nil {
			return e
		}
	}
	if count == 0 {
		return fmt.Errorf("no verified released observations")
	}
	if e = f.Sync(); e != nil {
		return e
	}
	current, e := os.ReadFile(*model)
	if e != nil {
		return e
	}
	if digest(current) != digest(mb) {
		return fmt.Errorf("teacher changed")
	}
	meta := map[string]any{"version": "hjepa_weapon_data_v1", "feature_version": policy.WeaponFeatureVersion, "split": *split, "rows": count, "seeds": seeds, "sources_sha256": sources, "native_receipts_sha256": receipts, "teacher_source": *model, "teacher_source_sha256": digest(mb), "teacher_argmax_histogram": hist, "label_kind": "hypothetical_u1_weapon_distribution_v1", "scope": "Client-only causal observations, actual past teacher memory, no future/server outcome inputs. Offline recommendations, not executed actions."}
	b, e := json.MarshalIndent(meta, "", "  ")
	if e != nil {
		return e
	}
	mf, e := os.OpenFile(*out+".meta.json", os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
	if e != nil {
		return e
	}
	defer mf.Close()
	if _, e = mf.Write(b); e != nil {
		return e
	}
	fmt.Printf("Exported %d weapon labels, %d seeds: %v\n", count, len(seeds), hist)
	return nil
}
func main() {
	if e := run(); e != nil {
		fmt.Fprintln(os.Stderr, e)
		os.Exit(1)
	}
}
