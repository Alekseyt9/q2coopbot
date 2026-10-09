// Export observed corrective aim queries from closed native-verified captures.
package main

import (
	"bufio"
	"compress/gzip"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"q2coopbot/internal/aimquery"
	"q2coopbot/internal/learningenv"
	"q2coopbot/internal/policy"
	"q2coopbot/internal/quake"
)

type member struct {
	Seed  int    `json:"seed"`
	Split string `json:"split"`
	Steps string `json:"steps"`
	SHA   string `json:"steps_sha256"`
}
type sample struct {
	Features    []float64                `json:"features"`
	Targets     [4]float64               `json:"targets"`
	Mask        [4]bool                  `json:"mask"`
	Attack      bool                     `json:"attack"`
	Identity    policy.Identity          `json:"identity"`
	Seed        int                      `json:"seed"`
	Step        int                      `json:"step"`
	Labels      []aimquery.TargetedLabel `json:"target_queries"`
	QueryMove   *quake.Vec3              `json:"query_move_label,omitempty"`
	QueryDucked *bool                    `json:"query_ducked_label,omitempty"`
}

func hash(path string) (string, error) {
	b, e := os.ReadFile(path)
	if e != nil {
		return "", e
	}
	h := sha256.Sum256(b)
	return hex.EncodeToString(h[:]), nil
}
func run() error {
	in := flag.String("spec", "", "Frozen native member steps and hashes")
	out := flag.String("out", "", "Fresh compressed observation/query dataset")
	center := flag.Bool("center-muzzle", false, "Verified center-hand blaster/aimfix0 instantaneous muzzle queries")
	postMove := flag.Bool("post-move-labels", false, "Offline labels conditional on achieved client-observed movement; inputs stay pre-command")
	flag.Parse()
	if *postMove && !*center {
		return fmt.Errorf("post-move labels require center-muzzle")
	}
	b, e := os.ReadFile(*in)
	if e != nil {
		return e
	}
	var spec struct {
		Members []member `json:"members"`
	}
	if e = json.Unmarshal(b, &spec); e != nil {
		return e
	}
	if len(spec.Members) == 0 {
		return fmt.Errorf("empty native corpus")
	}
	if _, e = os.Stat(*out); !os.IsNotExist(e) {
		return fmt.Errorf("fresh output required")
	}
	if e = os.MkdirAll(*out, 0755); e != nil {
		return e
	}
	counts := map[string]int{}
	aim := map[string]int{}
	masked := map[string]int{}
	seeds := map[int]string{}
	for _, split := range []string{"train", "validation"} {
		path := filepath.Join(*out, split+".jsonl.gz")
		f, e := os.Create(path)
		if e != nil {
			return e
		}
		gz := gzip.NewWriter(f)
		enc := json.NewEncoder(gz)
		for _, m := range spec.Members {
			if m.Split != split {
				continue
			}
			if _, ok := seeds[m.Seed]; ok {
				return fmt.Errorf("reused corpus seed")
			}
			seeds[m.Seed] = split
			h, e := hash(m.Steps)
			if e != nil {
				return e
			}
			if h != m.SHA {
				return fmt.Errorf("native steps changed")
			}
			input, e := os.Open(m.Steps)
			if e != nil {
				return e
			}
			scanner := bufio.NewScanner(input)
			scanner.Buffer(make([]byte, 65536), 4*1024*1024)
			for scanner.Scan() {
				var step learningenv.Step
				if e = json.Unmarshal(scanner.Bytes(), &step); e != nil {
					return e
				}
				o := step.Observation
				if o.Identity.Life != 1 || o.Identity.Frame <= 100 || o.Health <= 0 || o.AgeMS < 0 || o.AgeMS > 300 || step.Owner != "provider" {
					continue
				}
				if step.Version != learningenv.StepVersion {
					return fmt.Errorf("invalid native step version")
				}
				proven := step.Native != nil && step.Execution != nil && step.Execution.Matched && step.Execution.WindowExclusive && step.Execution.RecoveryCommands == 0
				x, e := policy.FeaturesForVersion(o, policy.TargetFeatureVersion)
				if e != nil {
					return e
				}
				labels, e := aimquery.Targeted(o)
				var queryMove *quake.Vec3
				var queryDucked *bool
				if *center {
					labels, e = aimquery.CenterMuzzle(o)
				}
				moveKnown := !*postMove
				if *postMove {
					labels = nil
					if step.Next != nil && step.Next.Health > 0 && policy.SameLife(o.Identity, step.Next.Identity) && step.Next.Identity.Frame == o.Identity.Frame+1 {
						delta := quake.Vec3{}
						for axis := range delta {
							delta[axis] = step.Next.Position[axis] - o.Position[axis]
							if math.IsNaN(delta[axis]) || math.IsInf(delta[axis], 0) {
								return fmt.Errorf("invalid observed movement label")
							}
						}
						post := o
						post.Enemies = append([]policy.Enemy(nil), o.Enemies...)
						for i := range post.Enemies {
							for axis := range delta {
								post.Enemies[i].Relative[axis] -= delta[axis]
							}
						}
						post.Ducked = step.Next.Ducked
						labels, e = aimquery.CenterMuzzle(post)
						for i := range labels {
							labels[i].Version = "observed_center_blaster_postmove_query_v3"
						}
						queryMove, queryDucked = &delta, &post.Ducked
						moveKnown = true
					}
				}
				if e != nil {
					return e
				}
				blocked := !proven || !moveKnown
				if !proven {
					masked[split]++
				}
				for _, v := range step.Interventions {
					if v.Component == "aim" || v.Component == "pitch" {
						blocked = true
					}
				}
				valid := false
				for _, v := range labels {
					valid = valid || v.LeadKnown && v.RecoilKnown
				}
				if e = enc.Encode(sample{Features: x, Identity: o.Identity, Seed: m.Seed, Step: step.Index, Labels: labels, QueryMove: queryMove, QueryDucked: queryDucked, Mask: [4]bool{false, !blocked && len(labels) > 0, false, false}}); e != nil {
					return e
				}
				counts[split]++
				if !blocked && valid {
					aim[split]++
				}
			}
			if e = scanner.Err(); e != nil {
				return e
			}
			input.Close()
		}
		if e = gz.Close(); e != nil {
			return e
		}
		if e = f.Close(); e != nil {
			return e
		}
		if counts[split] == 0 || aim[split] == 0 {
			return fmt.Errorf("empty split or aim labels")
		}
	}
	for _, m := range spec.Members {
		if m.Split != "train" && m.Split != "validation" {
			return fmt.Errorf("unsupported split")
		}
	}
	hashes := map[string]string{}
	for split := range counts {
		h, e := hash(filepath.Join(*out, split+".jsonl.gz"))
		if e != nil {
			return e
		}
		hashes[split] = h
	}
	specSHA, e := hash(*in)
	if e != nil {
		return e
	}
	report := map[string]any{"version": "combat_target_sequence_v1", "feature_version": policy.TargetFeatureVersion, "test_deferred": true, "counts": counts, "aim_rows": aim, "masked_unproven_context": masked, "data_sha256": hashes, "spec_sha256": specSHA, "seed_splits": seeds, "scope": "Own-policy native matched first-life context, actual prior target intent. Unproven final context remains masked. Unexecuted observed eye-origin blaster interception queries; machinegun recoil unknown, no hit/reward credit or optimal-target claim. Gzip streams."}
	report["target_query_version"] = aimquery.TargetedVersion
	if *center {
		report["target_query_version"] = aimquery.CenterMuzzleVersion
		report["scope"] = "Client-only center-hand blaster muzzle24forward/viewheight-8, aimfix0 instantaneous intercept queries. No firing delay, future movement, acceleration, hits or server position enter labels/features. Machinegun recoil unknown. Native proof and context masks preserved. Gzip streams."
	}
	if *postMove {
		report["target_query_version"] = "observed_center_blaster_postmove_query_v3"
		report["future_client_observation_labels"] = true
		report["future_model_inputs"] = false
		report["scope"] = "Offline center-hand blaster queries conditional on achieved prior-policy client-observed displacement/stance. Original observed target/velocity; no future target or server positions. Features remain pre-command V7/854. Teacher labels may use next client position, which is never a policy input. Factual movement is not a counterfactual corrected-angle rollout; native comparison required. Missing/dead/cross-life next context masked."
	}
	r, e := os.Create(filepath.Join(*out, "report.json"))
	if e != nil {
		return e
	}
	defer r.Close()
	return json.NewEncoder(r).Encode(report)
}
func main() {
	if e := run(); e != nil {
		fmt.Fprintln(os.Stderr, e)
		os.Exit(1)
	}
}
