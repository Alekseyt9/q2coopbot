// Query a frozen PPO ensemble offline on native TRAIN observations.
// Teacher recommendations are BC targets, never relabelled as executed actions.
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

func hash(b []byte) string { h := sha256.Sum256(b); return hex.EncodeToString(h[:]) }
func primitive(a policy.Action) []float64 {
	v := []float64{a.Forward, a.Side, a.YawDelta / 180, a.PitchDelta / 180, 0, 0, 0, 0}
	if a.Attack {
		v[4] = 1
	}
	switch a.Vertical {
	case "release":
		v[5] = 1
	case "jump":
		v[6] = 1
	case "crouch":
		v[7] = 1
	}
	return v
}
func run() error {
	root := flag.String("root", "", "native training capture root")
	model := flag.String("model", "", "frozen source PPO")
	out := flag.String("out", "", "fresh teacher JSONL")
	flag.Parse()
	if *root == "" || *model == "" || *out == "" {
		return fmt.Errorf("root,model,out required")
	}
	b, e := os.ReadFile(*model)
	if e != nil {
		return e
	}
	originalSHA := hash(b)
	var source map[string]any
	if e = json.Unmarshal(b, &source); e != nil {
		return e
	}
	// Joint aim candidates require the original stochastic execution contract.
	// Average sixteen source-policy samples instead of disabling that contract.
	var variants []string
	variantHashes := map[string]string{}
	for i := 0; i < 16; i++ {
		source["sampling_seed"] = 700000 + i
		variant, err := json.Marshal(source)
		if err != nil {
			return err
		}
		path := fmt.Sprintf("%s.teacher-%02d.json", *out, i)
		if err = writeFresh(path, variant); err != nil {
			return err
		}
		variants = append(variants, path)
		variantHashes[path] = hash(variant)
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
	f, e := os.OpenFile(*out, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
	if e != nil {
		return e
	}
	defer f.Close()
	enc := json.NewEncoder(f)
	sources := map[string]string{}
	receipts := map[string]string{}
	count := 0
	for _, path := range paths {
		var teachers []*policy.PPO
		for _, path := range variants {
			p, err := policy.LoadPPO(path)
			if err != nil {
				return err
			}
			teachers = append(teachers, p)
		}
		rp := filepath.Join(filepath.Dir(path), "report.json")
		rb, e := os.ReadFile(rp)
		if e != nil {
			return e
		}
		var r struct {
			Proof struct {
				Accepted bool `json:"accepted"`
			} `json:"command_proof"`
			Sync  bool `json:"synchronous_confirmed"`
			Reset bool `json:"observed_reset_confirmed"`
		}
		if e = json.Unmarshal(rb, &r); e != nil {
			return e
		}
		if !r.Proof.Accepted || !r.Sync || !r.Reset {
			return fmt.Errorf("unverified native teacher source")
		}
		receipts[rp] = hash(rb)
		b, e := os.ReadFile(path)
		if e != nil {
			return e
		}
		sources[path] = hash(b)
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
			// Advance teacher memory only through actual past observations in this life.
			var samples []policy.Action
			for _, p := range teachers {
				a, err := p.Decide(o)
				if err != nil {
					return err
				}
				if a.Weapon == "" || a.Weapon == "Machinegun" {
					samples = append(samples, a)
				}
			}
			if !teacherEligible(s) {
				continue
			}
			if len(samples) < 8 || len(o.Enemies) > 1 {
				continue
			}
			x, e := policy.FeaturesForVersion(o, policy.WeaponFeatureVersion)
			if e != nil {
				return e
			}
			next, e := policy.FeaturesForVersion(*s.Next, policy.WeaponFeatureVersion)
			if e != nil {
				return e
			}
			id := o.Identity
			id.Frame = 0
			identity, _ := json.Marshal(id)
			row := map[string]any{"episode": s.Episode + ":" + string(identity), "frame": o.Identity.Frame, "features": x, "next_features": next, "action": primitive(s.AppliedAction), "teacher_action": meanPrimitive(samples), "utility": 0, "label_kind": "hypothetical_ppo_ensemble_v1"}
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
		return fmt.Errorf("no native teacher observations")
	}
	if e = f.Sync(); e != nil {
		return e
	}
	meta := map[string]any{"version": "hjepa_data_v1", "feature_version": policy.WeaponFeatureVersion, "rows": count, "sources_sha256": sources, "native_receipts_sha256": receipts, "teacher_source_sha256": originalSHA, "teacher_variants_sha256": variantHashes, "label_kind": "hypothetical_ppo_ensemble_v1", "scope": "Mean of sixteen frozen PPO samples on single-enemy native observations; only sampling seeds differ. Actual action remains separate for world modelling. Guarded states may be labelled; recommendations were not executed. No future/server effect inputs."}
	mb, _ := json.MarshalIndent(meta, "", "  ")
	if e = writeFresh(*out+".meta.json", mb); e != nil {
		return e
	}
	fmt.Printf("Exported %d hypothetical teacher labels\n", count)
	return nil
}

// Eligibility follows verified native dispatch, not a guessed client frame.
// A native game release at game_frame=100 can first appear at client frame=98.
func teacherEligible(s learningenv.Step) bool {
	o := s.Observation
	return s.Next != nil && policy.SameLife(o.Identity, s.Next.Identity) && s.Next.Identity.Frame == o.Identity.Frame+1 &&
		s.Execution != nil && s.Execution.Matched && s.Execution.WindowExclusive &&
		s.Native != nil && s.Native.Spawncount == o.Identity.Spawncount && s.Native.Actor == o.Identity.Actor &&
		s.Native.EndFrame == s.Native.BeginFrame+1 && !s.Terminal && !s.Truncated && s.Owner == "provider"
}
func meanPrimitive(samples []policy.Action) []float64 {
	mean := make([]float64, 8)
	sin, cos := 0., 0.
	for _, a := range samples {
		v := primitive(a)
		for i, x := range v {
			mean[i] += x / float64(len(samples))
		}
		angle := a.YawDelta * math.Pi / 180
		sin += math.Sin(angle)
		cos += math.Cos(angle)
	}
	mean[2] = math.Atan2(sin, cos) / math.Pi
	return mean
}
func writeFresh(path string, b []byte) error {
	f, e := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
	if e != nil {
		return e
	}
	defer f.Close()
	_, e = f.Write(b)
	return e
}
func main() {
	if e := run(); e != nil {
		fmt.Fprintln(os.Stderr, e)
		os.Exit(1)
	}
}
