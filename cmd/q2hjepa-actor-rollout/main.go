// Export native stochastic actor proposals with exact replay and raw latents.
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
	"reflect"
	"strings"
)

func digest(b []byte) string { h := sha256.Sum256(b); return hex.EncodeToString(h[:]) }
func run() error {
	root := flag.String("root", "", "completed non-diagnostic TRAIN matrix")
	model := flag.String("model", "", "frozen stochastic H-JEPA actor")
	variant := flag.String("variant", "", "matrix variant")
	out := flag.String("out", "", "fresh JSONL")
	flag.Parse()
	if *root == "" || *model == "" || *out == "" || *variant == "" || filepath.Base(*variant) != *variant {
		return fmt.Errorf("root,model,variant,out required")
	}
	for _, p := range []*string{root, model, out} {
		absolute, err := filepath.Abs(*p)
		if err != nil {
			return err
		}
		*p = absolute
	}
	for _, name := range []string{*out, *out + ".meta.json"} {
		if _, err := os.Stat(name); err == nil {
			return fmt.Errorf("fresh export required")
		} else if !os.IsNotExist(err) {
			return err
		}
	}
	sources := map[string]string{}
	read := func(path string) ([]byte, error) {
		b, e := os.ReadFile(path)
		if e == nil {
			sources[path] = digest(b)
		}
		return b, e
	}
	mb, err := read(*model)
	if err != nil {
		return err
	}
	var spec policy.HJEPAFile
	if err = json.Unmarshal(mb, &spec); err != nil {
		return err
	}
	p, err := policy.LoadHJEPA(*model)
	if err != nil {
		return err
	}
	if spec.PrimitiveSampling == nil {
		return fmt.Errorf("stochastic actor model required")
	}
	b, err := read(filepath.Join(*root, "complete.json"))
	if err != nil {
		return err
	}
	var done struct {
		State      string `json:"state"`
		Diagnostic bool   `json:"diagnostic_weapon_branches"`
		PlanSHA    string `json:"source_plan_sha256"`
		Models     []struct {
			Name string `json:"name"`
			SHA  string `json:"sha256"`
		} `json:"models"`
	}
	if err = json.Unmarshal(b, &done); err != nil {
		return err
	}
	if done.State != "complete" || done.Diagnostic {
		return fmt.Errorf("completed non-diagnostic matrix required")
	}
	bound := false
	for _, v := range done.Models {
		if v.Name == *variant && v.SHA == digest(mb) {
			bound = true
		}
	}
	if !bound {
		return fmt.Errorf("actor not bound to matrix")
	}
	b, err = read(filepath.Join(*root, "plan.json"))
	if err != nil {
		return err
	}
	if digest(b) != done.PlanSHA {
		return fmt.Errorf("source plan changed")
	}
	paths, err := filepath.Glob(filepath.Join(*root, *variant, "case-*", "worker-*", "dataset", "steps.jsonl"))
	if err != nil {
		return err
	}
	if len(paths) == 0 {
		return fmt.Errorf("no native actor captures")
	}
	rows := []map[string]any{}
	episodes := []map[string]any{}
	seen := map[int]bool{}
	for _, path := range paths {
		worker := filepath.Dir(filepath.Dir(path))
		caseDir := filepath.Dir(worker)
		rb, err := read(filepath.Join(caseDir, "report.json"))
		if err != nil {
			return err
		}
		var report struct {
			Valid    bool              `json:"provenance_valid"`
			Complete bool              `json:"capture_complete"`
			Results  []json.RawMessage `json:"results"`
		}
		if err = json.Unmarshal(rb, &report); err != nil {
			return err
		}
		if !report.Valid || !report.Complete {
			return fmt.Errorf("invalid case proof")
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
				Start struct {
					Confirmed bool `json:"confirmed"`
				} `json:"generated_start"`
			}
			if err = json.Unmarshal(raw, &r); err != nil {
				return err
			}
			if strings.EqualFold(filepath.Clean(worker), filepath.Clean(r.Root)) {
				if !r.Valid || r.Fixture.Split != "train" || !r.Start.Confirmed {
					return fmt.Errorf("verified TRAIN source required")
				}
				result = raw
				seed = r.Seed
			}
		}
		if result == nil || seen[seed] {
			return fmt.Errorf("missing/duplicate source binding")
		}
		seen[seed] = true
		pb, err := read(filepath.Join(filepath.Dir(path), "report.json"))
		if err != nil {
			return err
		}
		var proof struct {
			Proof struct {
				Accepted bool `json:"accepted"`
				Recovery int  `json:"recovery_commands"`
			} `json:"command_proof"`
			Sync  bool `json:"synchronous_confirmed"`
			Reset bool `json:"observed_reset_confirmed"`
		}
		if err = json.Unmarshal(pb, &proof); err != nil {
			return err
		}
		if !proof.Proof.Accepted || proof.Proof.Recovery != 0 || !proof.Sync || !proof.Reset {
			return fmt.Errorf("native command/reset proof missing")
		}
		b, err := read(path)
		if err != nil {
			return err
		}
		scan := bufio.NewScanner(strings.NewReader(string(b)))
		scan.Buffer(make([]byte, 65536), 4*1024*1024)
		p.ResetEpisode()
		count := 0
		last := policy.Identity{}
		for scan.Scan() {
			var s learningenv.Step
			if err = json.Unmarshal(scan.Bytes(), &s); err != nil {
				return err
			}
			if s.Owner != "provider" || s.Observation.Health <= 0 {
				continue
			}
			o := s.Observation
			if s.Provider != p.Version() || s.Sample == nil || s.Native == nil || s.Execution == nil || !s.Execution.Matched || s.Native.EndFrame != s.Native.BeginFrame+1 || s.Native.Actor != o.Identity.Actor || s.Native.Spawncount != o.Identity.Spawncount {
				return fmt.Errorf("actor sample or native join missing")
			}
			if count > 0 && (!policy.SameLife(last, o.Identity) || o.Identity.Frame <= last.Frame) {
				return fmt.Errorf("life/frame order changed")
			}
			last = o.Identity
			action, err := p.Decide(o)
			if err != nil {
				return err
			}
			if action != s.Action || !reflect.DeepEqual(p.LastSample(), s.Sample) {
				return fmt.Errorf("actor behavior replay mismatch")
			}
			lp, err := p.PrimitiveLogProbability(o, *s.Sample)
			if err != nil {
				return err
			}
			if math.IsNaN(s.Sample.LogProbability) || math.IsInf(s.Sample.LogProbability, 0) || math.Abs(lp-s.Sample.LogProbability) > 1e-9 {
				return fmt.Errorf("actor behavior likelihood mismatch")
			}
			features, err := policy.HJEPAFeatures(o, spec.Features)
			if err != nil {
				return err
			}
			_, raw, _, err := p.Replay(features, []float64{0, 0, 0, 0, 0, 1, 0, 0})
			if err != nil {
				return err
			}
			weaponLP := 0.
			var weaponProbs []float64
			if spec.WeaponSamplingSeed != nil {
				weaponProbs, err = p.WeaponDistribution(o)
				if err != nil {
					return err
				}
				weaponLP = math.Log(weaponProbs[s.Sample.Weapon])
			}
			rows = append(rows, map[string]any{"version": "hjepa_actor_rollout_row_v1", "seed": seed, "split": "train", "identity": o.Identity, "features": features, "sample": s.Sample, "actor_raw": raw, "behavior_log_probability": lp, "primitive_log_probability": lp - weaponLP, "weapon_probabilities": weaponProbs, "source": path, "source_sha256": sources[path], "step_index": s.Index, "exclusive": s.Execution.WindowExclusive})
			count++
		}
		if err = scan.Err(); err != nil {
			return err
		}
		if count == 0 {
			return fmt.Errorf("no actor choices")
		}
		episodes = append(episodes, map[string]any{"seed": seed, "rows": count, "native_result": result})
	}
	f, err := os.OpenFile(*out, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
	if err != nil {
		return err
	}
	defer f.Close()
	encoder := json.NewEncoder(f)
	for _, row := range rows {
		if err = encoder.Encode(row); err != nil {
			return err
		}
	}
	if err = f.Sync(); err != nil {
		return err
	}
	b, err = os.ReadFile(*out)
	if err != nil {
		return err
	}
	meta := map[string]any{"version": "hjepa_onpolicy_actor_rollout_v1", "split": "train", "feature_version": spec.Features, "distribution": policy.HJEPAActorDistribution, "rows": len(rows), "data_sha256": digest(b), "episodes": episodes, "sources_sha256": sources, "model": *model, "model_sha256": digest(mb), "variant": *variant, "scope": "Actual full primitive Gaussian/categorical actor proposals with raw latent receipts, exact frozen replay, joint weapon likelihood and matched native dispatch. Features only are neural inputs; sampled latents, old raw logits/probabilities, outcome records and source identities are training labels/proofs. Guards remain environment; terminal nonexclusive rows marked, post-endpoint credit must be masked by native ledger. Not expert demonstrations or PPO-provider format."}
	metaBytes, err := json.MarshalIndent(meta, "", "  ")
	if err != nil {
		return err
	}
	mf, err := os.OpenFile(*out+".meta.json", os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
	if err != nil {
		return err
	}
	defer mf.Close()
	if _, err = mf.Write(metaBytes); err != nil {
		return err
	}
	fmt.Printf("Exported %d actor proposals from %d TRAIN worlds\n", len(rows), len(episodes))
	return nil
}
func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
