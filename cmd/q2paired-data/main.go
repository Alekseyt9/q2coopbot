// Experimental paired learner adapter. Numerical NN work is CUDA-only.
package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strings"

	"q2coopbot/internal/learningenv"
	"q2coopbot/internal/policy"
	"q2coopbot/internal/trainingepisodes"
)

func read(path string, target any) error {
	b, e := os.ReadFile(path)
	if e != nil {
		return e
	}
	return json.Unmarshal(b, target)
}
func rows[T any](path string) ([]T, error) {
	f, e := os.Open(path)
	if e != nil {
		return nil, e
	}
	defer f.Close()
	s := bufio.NewScanner(f)
	s.Buffer(make([]byte, 65536), 8*1024*1024)
	var result []T
	for s.Scan() {
		var row T
		if e = json.Unmarshal(s.Bytes(), &row); e != nil {
			return nil, e
		}
		result = append(result, row)
	}
	return result, s.Err()
}
func main() {
	if e := run(); e != nil {
		fmt.Fprintln(os.Stderr, e)
		os.Exit(1)
	}
}
func run() error {
	root := flag.String("case", "", "Closed paired capture")
	dataset := flag.String("dataset", "", "Verified v11 learner dataset")
	model := flag.String("model", "", "Frozen common behavior weights")
	exporter := flag.String("exporter", "", "Frozen q2combat-export binary")
	out := flag.String("out", "", "Fresh native CUDA-pending output")
	allowHorizon := flag.Bool("allow-horizon", false, "Include verified surviving horizon cases; horizon is not a victory terminal")
	flag.Parse()
	for _, p := range []*string{root, dataset, model, exporter, out} {
		if *p == "" {
			return fmt.Errorf("case,dataset,model,exporter,out required")
		}
		v, e := filepath.Abs(*p)
		if e != nil {
			return e
		}
		*p = v
	}
	if _, e := os.Stat(*out); !os.IsNotExist(e) {
		return fmt.Errorf("fresh output required")
	}
	var capture struct {
		Accepted     bool   `json:"accepted"`
		Source       bool   `json:"source_unchanged"`
		NativeSource bool   `json:"native_source_unchanged"`
		Release      bool   `json:"fixed_release_verified"`
		RNG          bool   `json:"post_frame_seed_verified"`
		Placements   bool   `json:"player_placements_verified"`
		World        bool   `json:"fixed_world_hold_verified"`
		LiveStop     bool   `json:"live_joint_stop_verified"`
		Seed         int    `json:"seed"`
		StopFrame    int    `json:"stop_frame"`
		Pairs        int    `json:"completed_pairs"`
		ModelSHA     string `json:"source_model_sha256"`
		ProviderSHA  string `json:"provider_sha256"`
		ServerSHA    string `json:"server_sha256"`
		ClientSHA    string `json:"client_sha256"`
		GameSHA      string `json:"game_sha256"`
	}
	if e := read(filepath.Join(*root, "report.json"), &capture); e != nil {
		return e
	}
	if !capture.Accepted || !capture.Source || !capture.NativeSource || !capture.Release || !capture.RNG || !capture.Placements || !capture.World || !capture.LiveStop && !*allowHorizon {
		return fmt.Errorf("closed fixed paired live-death capture required for adapter pilot")
	}
	sources := map[string]string{}
	pin := func(path string, expected string) error {
		path, e := filepath.Abs(path)
		if e != nil {
			return e
		}
		h, e := trainingepisodes.Hash(path)
		if e != nil {
			return e
		}
		if expected != "" && !strings.EqualFold(h, expected) {
			return fmt.Errorf("source changed: %s", path)
		}
		sources[path] = h
		return nil
	}
	for path, h := range map[string]string{*model: capture.ModelSHA, filepath.Join(*root, "seeded-weights.json"): capture.ProviderSHA, filepath.Join(*root, "q2coopbot.exe"): capture.ClientSHA, filepath.Join(*root, "runtime", "q2ded.exe"): capture.ServerSHA, filepath.Join(*root, "runtime", "baseq2", "game.dll"): capture.GameSHA, filepath.Join(*root, "report.json"): "", *exporter: ""} {
		if e := pin(path, h); e != nil {
			return e
		}
	}
	behavior, e := policy.LoadPPO(*model)
	if e != nil {
		return e
	}
	seeded, e := policy.LoadPPO(filepath.Join(*root, "seeded-weights.json"))
	if e != nil {
		return e
	}
	if !behavior.IsStochastic() || !seeded.IsStochastic() || behavior.Version() != seeded.Version() || seeded.SamplingSeed() != int64(capture.Seed) {
		return fmt.Errorf("behavior weights/seed mismatch")
	}
	var report struct {
		Paired       bool                             `json:"paired_confirmed"`
		Role         int                              `json:"paired_role"`
		Experimental bool                             `json:"paired_experimental_reward"`
		Reset        bool                             `json:"observed_reset_confirmed"`
		Synchronous  bool                             `json:"synchronous_confirmed"`
		Reward       learningenv.RewardConfig         `json:"reward_config"`
		Boundary     *learningenv.PairedDeathBoundary `json:"joint_death_boundary"`
		Sources      map[string]string                `json:"paired_source_sha256"`
	}
	if e := read(filepath.Join(*dataset, "report.json"), &report); e != nil {
		return e
	}
	if !report.Paired || report.Role != 0 || !report.Experimental || !report.Reset || !report.Synchronous || report.Reward.Version != learningenv.CoopRewardVersion {
		return fmt.Errorf("verified v11 primary learner dataset with joint terminal required")
	}
	end := "surviving_horizon"
	if capture.LiveStop {
		if report.Boundary == nil || report.Boundary.Seed != capture.Seed || report.Boundary.EndFrame != capture.StopFrame {
			return fmt.Errorf("live joint boundary mismatch")
		}
		end = "joint_death"
	} else if report.Boundary != nil {
		return fmt.Errorf("unverified live joint boundary")
	}
	for path, h := range report.Sources {
		if e := pin(path, h); e != nil {
			return e
		}
	}
	for _, name := range []string{"report.json", "steps.jsonl", "rewards.jsonl", "server_outcomes.jsonl", "reward-config.json"} {
		if e := pin(filepath.Join(*dataset, name), ""); e != nil {
			return e
		}
	}
	steps, e := rows[learningenv.Step](filepath.Join(*dataset, "steps.jsonl"))
	if e != nil {
		return e
	}
	if len(steps) == 0 {
		return fmt.Errorf("empty paired dataset")
	}
	if e := os.MkdirAll(*out, 0755); e != nil {
		return e
	}
	replay := filepath.Join(*out, "replay")
	args := []string{"--trace", filepath.Join(*root, "PairLearner.jsonl"), "--paired-peer-trace", filepath.Join(*root, "PairLeader.jsonl"), "--paired-role", "0", "--paired-stop-on-death", "--paired-experimental-reward", "--reward-config", filepath.Join(*dataset, "reward-config.json"), "--server-log", filepath.Join(*root, "server.log"), "--client-name", "PairLearner", "--reset-expectation", filepath.Join(*root, "reset-role-0.json"), "--synchronous", "--require-execution", "--worker", steps[0].Worker, "--episode", steps[0].Episode, "--out", replay}
	output, e := exec.Command(*exporter, args...).CombinedOutput()
	if e != nil {
		return fmt.Errorf("paired replay: %w %s", e, output)
	}
	for _, name := range []string{"steps.jsonl", "rewards.jsonl", "server_outcomes.jsonl"} {
		a, e := trainingepisodes.Hash(filepath.Join(replay, name))
		if e != nil {
			return e
		}
		b, e := trainingepisodes.Hash(filepath.Join(*dataset, name))
		if e != nil {
			return e
		}
		if a != b {
			return fmt.Errorf("paired replay differs: %s", name)
		}
		if e := pin(filepath.Join(replay, name), a); e != nil {
			return e
		}
	}
	rewards, e := rows[learningenv.Reward](filepath.Join(replay, "rewards.jsonl"))
	if e != nil {
		return e
	}
	if len(rewards) != len(steps) {
		return fmt.Errorf("paired reward count mismatch")
	}
	var data, context bytes.Buffer
	enc, ctx := json.NewEncoder(&data), json.NewEncoder(&context)
	count, contexts, terminals := 0, 0, 0
	lastHandoff := false
	for i, s := range steps {
		if s.Owner != "provider" {
			continue
		} // Scripted peer/rules actions never train.
		if s.Sample == nil || s.Provider != behavior.Version() || s.Sample.Version != behavior.Version() || s.Sample.SamplingSeed != int64(capture.Seed) || s.Observation.Identity.Life != 1 {
			return fmt.Errorf("missing/foreign learner sample")
		}
		a, e := seeded.ActionForSample(s.Observation, *s.Sample)
		if e != nil {
			return e
		}
		if !reflect.DeepEqual(a, s.Action) {
			return fmt.Errorf("altered sampled learner action")
		}
		x, e := policy.FeaturesForVersion(s.Observation, behavior.FeatureVersion())
		if e != nil {
			return e
		}
		if behavior.IsRecurrent() {
			if e = ctx.Encode(map[string]any{"features": x, "seed": capture.Seed, "index": s.Index, "frame": s.Observation.Identity.Frame, "memory": s.Sample.Memory}); e != nil {
				return e
			}
			contexts++
		}
		r := rewards[i]
		if !r.Available || r.Score == nil || s.Next == nil {
			continue
		}
		if r.Step != s.Index || r.Worker != s.Worker || r.Episode != s.Episode {
			return fmt.Errorf("reward/sample identity differs")
		}
		nx, e := policy.FeaturesForVersion(*s.Next, behavior.FeatureVersion())
		if e != nil {
			return e
		}
		if s.Next.Identity.Frame != s.Observation.Identity.Frame+1 {
			return fmt.Errorf("nonconsecutive learner transition")
		}
		zero := s.Terminal || s.Truncated && s.Reason == "control_handoff"
		lastHandoff = s.Truncated && s.Reason == "control_handoff"
		if s.Terminal {
			terminals++
		}
		exported := map[string]any{"features": x, "next_features": nx, "next_value": 0, "sample": s.Sample, "reward": *r.Score, "terminal": s.Terminal, "truncated": s.Truncated, "seed": capture.Seed, "index": s.Index, "frame": s.Observation.Identity.Frame, "next_frame": s.Next.Identity.Frame, "interventions": s.Interventions, "next_reset": false, "bootstrap_zero": zero}
		if e = enc.Encode(exported); e != nil {
			return e
		}
		count++
	}
	if capture.LiveStop && terminals == 0 && lastHandoff {
		end = "joint_death_after_handoff"
	}
	if count == 0 || capture.LiveStop && terminals != 1 && !lastHandoff || !capture.LiveStop && terminals != 0 || terminals > 1 {
		return fmt.Errorf("pilot learner terminal does not match capture outcome")
	}
	for path, h := range sources {
		if e := pin(path, h); e != nil {
			return e
		}
	}
	rollout := filepath.Join(*out, "native-rollout.jsonl")
	if e := os.WriteFile(rollout, data.Bytes(), 0644); e != nil {
		return e
	}
	h, e := trainingepisodes.Hash(rollout)
	if e != nil {
		return e
	}
	rewardSHA, e := trainingepisodes.Hash(filepath.Join(*dataset, "reward-config.json"))
	if e != nil {
		return e
	}
	meta := map[string]any{"version": "combat_ppo_native_cuda_pending_v1", "numerical_verification": "cuda_pending", "feature_version": behavior.FeatureVersion(), "policy_version": behavior.Version(), "model_sha256": sources[*model], "reward_config_sha256": rewardSHA, "reward_version": report.Reward.Version, "aim_gamma": report.Reward.AimGamma, "training_monster_health": 30, "rows": count, "skipped": len(steps) - count, "terminals": terminals, "native_rollout_sha256": h, "source_sha256": sources, "paired_adapter_version": "coop_primary_native_adapter_v1", "paired_adapter_verified": true, "paired_training_ready": false, "scope": "Experimental primary learner at one base1 Soldier/Blaster site with scripted moving peer. Native replay and sampled action contract verified; CUDA likelihood/value/memory/bootstrap verification required. Full world reset equivalence unproven."}
	meta["paired_episode_end"] = end
	if behavior.IsRecurrent() {
		p := filepath.Join(*out, "sequence.jsonl")
		if e := os.WriteFile(p, context.Bytes(), 0644); e != nil {
			return e
		}
		h, e := trainingepisodes.Hash(p)
		if e != nil {
			return e
		}
		meta["recurrent_version"], meta["sequence_rows"], meta["sequence_sha256"] = behavior.MemoryVersion(), contexts, h
	}
	b, e := json.MarshalIndent(meta, "", "  ")
	if e != nil {
		return e
	}
	if e = os.WriteFile(filepath.Join(*out, "report.json"), b, 0644); e != nil {
		return e
	}
	fmt.Printf("Paired learner rows=%d context=%d terminals=%d; CUDA pending\n", count, contexts, terminals)
	return nil
}
