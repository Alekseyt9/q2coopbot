// Export actual stochastic H-JEPA choices, never hypothetical branch labels.
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
	"q2coopbot/internal/learningenv"
	"q2coopbot/internal/policy"
	"strings"
)

func digest(b []byte) string { h := sha256.Sum256(b); return hex.EncodeToString(h[:]) }
func run() error {
	root := flag.String("root", "", "completed non-diagnostic matrix")
	model := flag.String("model", "", "frozen stochastic behavior model")
	variant := flag.String("variant", "", "matrix variant")
	out := flag.String("out", "", "fresh JSONL")
	flag.Parse()
	if *root == "" || *model == "" || *variant == "" || *out == "" || filepath.Base(*variant) != *variant {
		return fmt.Errorf("root,model,variant,out required")
	}
	sources := map[string]string{}
	read := func(path string) ([]byte, error) {
		b, e := os.ReadFile(path)
		if e == nil {
			sources[path] = digest(b)
		}
		return b, e
	}
	mb, e := read(*model)
	if e != nil {
		return e
	}
	var spec policy.HJEPAFile
	if e = json.Unmarshal(mb, &spec); e != nil {
		return e
	}
	if spec.WeaponSamplingSeed == nil || spec.WeaponOptionFrames != 0 || spec.WeaponFeatures != policy.HJEPAWeaponFeatureVersion {
		return fmt.Errorf("frame-wise stochastic context weapon model required")
	}
	p, e := policy.LoadHJEPA(*model)
	if e != nil {
		return e
	}
	cb, e := read(filepath.Join(*root, "complete.json"))
	if e != nil {
		return e
	}
	var complete struct {
		State      string `json:"state"`
		Diagnostic bool   `json:"diagnostic_weapon_branches"`
		Models     []struct {
			Name string `json:"name"`
			SHA  string `json:"sha256"`
		} `json:"models"`
	}
	if e = json.Unmarshal(cb, &complete); e != nil {
		return e
	}
	if complete.State != "complete" || complete.Diagnostic {
		return fmt.Errorf("non-diagnostic completed matrix required")
	}
	admitted := false
	for _, v := range complete.Models {
		if v.Name == *variant && v.SHA == digest(mb) {
			admitted = true
		}
	}
	if !admitted {
		return fmt.Errorf("behavior model not bound to matrix")
	}
	paths, e := filepath.Glob(filepath.Join(*root, *variant, "case-*", "worker-*", "dataset", "steps.jsonl"))
	if e != nil {
		return e
	}
	if len(paths) == 0 {
		return fmt.Errorf("no captures")
	}
	rows := []map[string]any{}
	episodes := []map[string]any{}
	seeds := map[int]bool{}
	names := []string{"", "Blaster", "Shotgun", "Super Shotgun", "Machinegun", "Chaingun", "Grenade Launcher", "Rocket Launcher", "HyperBlaster", "Railgun", "BFG10K", "Grenades"}
	for _, path := range paths {
		worker := filepath.Dir(filepath.Dir(path))
		caseDir := filepath.Dir(worker)
		rb, e := read(filepath.Join(caseDir, "report.json"))
		if e != nil {
			return e
		}
		var report struct {
			Valid    bool              `json:"provenance_valid"`
			Complete bool              `json:"capture_complete"`
			Results  []json.RawMessage `json:"results"`
		}
		if e = json.Unmarshal(rb, &report); e != nil {
			return e
		}
		if !report.Valid || !report.Complete {
			return fmt.Errorf("invalid case receipt")
		}
		var result json.RawMessage
		seed := 0
		for _, raw := range report.Results {
			var r struct {
				Root    string `json:"root"`
				Seed    int    `json:"seed"`
				Valid   bool   `json:"capture_valid"`
				Fixture struct {
					Split string `json:"split"`
				} `json:"generated_fixture"`
			}
			if e = json.Unmarshal(raw, &r); e != nil {
				return e
			}
			if strings.EqualFold(filepath.Clean(r.Root), filepath.Clean(worker)) {
				if !r.Valid || r.Fixture.Split != "train" {
					return fmt.Errorf("only verified TRAIN admitted")
				}
				result = raw
				seed = r.Seed
			}
		}
		if result == nil || seeds[seed] {
			return fmt.Errorf("missing/duplicate worker binding")
		}
		seeds[seed] = true
		pb, e := read(filepath.Join(filepath.Dir(path), "report.json"))
		if e != nil {
			return e
		}
		var proof struct {
			Proof struct {
				Accepted bool `json:"accepted"`
				Recovery int  `json:"recovery_commands"`
			} `json:"command_proof"`
			Sync  bool `json:"synchronous_confirmed"`
			Reset bool `json:"observed_reset_confirmed"`
		}
		if e = json.Unmarshal(pb, &proof); e != nil {
			return e
		}
		if !proof.Proof.Accepted || proof.Proof.Recovery != 0 || !proof.Sync || !proof.Reset {
			return fmt.Errorf("native command/reset proof missing")
		}
		b, e := read(path)
		if e != nil {
			return e
		}
		scan := bufio.NewScanner(strings.NewReader(string(b)))
		scan.Buffer(make([]byte, 65536), 4*1024*1024)
		p.ResetEpisode()
		count := 0
		last := policy.Identity{}
		for scan.Scan() {
			var s learningenv.Step
			if e = json.Unmarshal(scan.Bytes(), &s); e != nil {
				return e
			}
			if s.Owner != "provider" || s.Observation.Health <= 0 {
				continue
			}
			o := s.Observation
			if s.Provider != p.Version() || s.Native == nil || s.Execution == nil || !s.Execution.Matched {
				return fmt.Errorf("source or dispatch mismatch: %s frame %d", path, o.Identity.Frame)
			}
			if count > 0 && (!policy.SameLife(last, o.Identity) || o.Identity.Frame <= last.Frame) {
				return fmt.Errorf("life/order mismatch")
			}
			last = o.Identity
			a, e := p.Decide(o)
			if e != nil {
				return e
			}
			if a != s.Action {
				return fmt.Errorf("behavior replay mismatch")
			}
			probs, e := p.WeaponDistribution(o)
			if e != nil {
				return e
			}
			slot := -1
			for i, name := range names {
				if name == a.Weapon {
					slot = i
				}
			}
			if slot < 0 || probs[slot] <= 0 {
				return fmt.Errorf("unavailable choice")
			}
			x, e := policy.HJEPAWeaponFeatures(o)
			if e != nil {
				return e
			}
			mask, e := policy.WeaponAvailability(o)
			if e != nil {
				return e
			}
			rows = append(rows, map[string]any{"seed": seed, "split": "train", "identity": o.Identity, "features": x, "availability": mask, "slot": slot, "behavior_probabilities": probs, "behavior_log_probability": math.Log(probs[slot]), "source": path, "step_index": s.Index, "exclusive": s.Execution.WindowExclusive})
			count++
		}
		if e = scan.Err(); e != nil {
			return e
		}
		if count == 0 {
			return fmt.Errorf("no actual choices")
		}
		episodes = append(episodes, map[string]any{"seed": seed, "rows": count, "native_result": result})
	}
	f, e := os.OpenFile(*out, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
	if e != nil {
		return e
	}
	defer f.Close()
	enc := json.NewEncoder(f)
	for _, r := range rows {
		if e = enc.Encode(r); e != nil {
			return e
		}
	}
	if e = f.Sync(); e != nil {
		return e
	}
	data, e := os.ReadFile(*out)
	if e != nil {
		return e
	}
	meta := map[string]any{"version": "hjepa_onpolicy_weapon_rollout_v1", "split": "train", "rows": len(rows), "data_sha256": digest(data), "episodes": episodes, "sources_sha256": sources, "model": *model, "model_sha256": digest(mb), "variant": *variant, "scope": "Actual categorical provider proposals with exact frozen replay and matched native dispatch; guards are part of environment. Outcomes are separate episode labels, never inputs. Terminal non-exclusive dispatch is retained, not credited as exclusive."}
	b, e := json.MarshalIndent(meta, "", "  ")
	if e != nil {
		return e
	}
	mf, e := os.OpenFile(*out+".meta.json", os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
	if e != nil {
		return e
	}
	defer mf.Close()
	_, e = mf.Write(b)
	if e == nil {
		fmt.Printf("%d choices in %d TRAIN episodes\n", len(rows), len(episodes))
	}
	return e
}
func main() {
	if e := run(); e != nil {
		fmt.Fprintln(os.Stderr, e)
		os.Exit(1)
	}
}
