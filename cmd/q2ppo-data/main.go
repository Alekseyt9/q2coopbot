// Re-export native proof before preparing one frozen on-policy PPO batch.
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
	"os/exec"
	"path/filepath"
	"q2coopbot/internal/learningenv"
	"q2coopbot/internal/policy"
	"q2coopbot/internal/trainingepisodes"
	"reflect"
	"strings"
)

func sha(path string) (string, error) {
	b, e := os.ReadFile(path)
	if e != nil {
		return "", e
	}
	h := sha256.Sum256(b)
	return hex.EncodeToString(h[:]), nil
}
func read(path string, v any) error {
	b, e := os.ReadFile(path)
	if e != nil {
		return e
	}
	return json.Unmarshal(b, v)
}
func rows[T any](path string) ([]T, error) {
	f, e := os.Open(path)
	if e != nil {
		return nil, e
	}
	defer f.Close()
	s := bufio.NewScanner(f)
	s.Buffer(make([]byte, 65536), 4*1024*1024)
	var result []T
	for s.Scan() {
		var x T
		if e = json.Unmarshal(s.Bytes(), &x); e != nil {
			return nil, e
		}
		result = append(result, x)
	}
	return result, s.Err()
}

type row struct {
	Features      []float64             `json:"features"`
	NextValue     float64               `json:"next_value"`
	Sample        policy.Sample         `json:"sample"`
	Reward        float64               `json:"reward"`
	Terminal      bool                  `json:"terminal"`
	Truncated     bool                  `json:"truncated"`
	Seed          int                   `json:"seed"`
	Index         int                   `json:"index"`
	Frame         int                   `json:"frame"`
	NextFrame     int                   `json:"next_frame"`
	Interventions []policy.Intervention `json:"interventions"`
	NextFeatures  []float64             `json:"next_features,omitempty"`
	NextReset     bool                  `json:"next_reset"`
	BootstrapZero bool                  `json:"bootstrap_zero"`
}

func main() {
	if e := run(); e != nil {
		fmt.Fprintln(os.Stderr, e)
		os.Exit(1)
	}
}
func run() error {
	batch := flag.String("batch", "", "fresh synchronous learned PPO batch")
	merge := flag.String("merge", "", "comma-separated verified rollout directories from the same frozen policy")
	out := flag.String("out", "", "fresh directory")
	model := flag.String("model", "", "frozen original PPO model")
	cudaOnly := flag.Bool("cuda-only", false, "native/features only; require CUDA finalization before merge or training")
	flag.Parse()
	if *merge != "" {
		return trainingepisodes.MergeRollouts(strings.Split(*merge, ","), *out)
	}
	var manifest struct {
		Provider       string                   `json:"provider"`
		Kind           string                   `json:"provider_kind"`
		Synchronous    bool                     `json:"synchronous"`
		RewardSHA      string                   `json:"reward_config_sha256"`
		ExporterSHA    string                   `json:"exporter_sha256"`
		ModelSHA       string                   `json:"model_weights_sha256"`
		Reward         learningenv.RewardConfig `json:"reward_config"`
		TrainingHealth int                      `json:"training_monster_health"`
		PostFrameRNG   bool                     `json:"post_frame_rng_reset"`
		ReleaseFrame   int                      `json:"release_game_frame"`
		Loadout        string                   `json:"loadout"`
		StopOnGoal     bool                     `json:"stop_on_goal"`
	}
	if e := read(filepath.Join(*batch, "manifest.json"), &manifest); e != nil {
		return e
	}
	if manifest.Provider != "learned" || manifest.Kind != policy.PPOKind || !manifest.Synchronous {
		return fmt.Errorf("fresh direct synchronous PPO batch required")
	}
	modelSHA, e := sha(*model)
	if e != nil {
		return e
	}
	if !strings.EqualFold(modelSHA, manifest.ModelSHA) {
		return fmt.Errorf("wrong behavior model")
	}
	behavior, e := policy.LoadPPO(*model)
	if e != nil {
		return e
	}
	if !behavior.IsStochastic() {
		return fmt.Errorf("deterministic rollout cannot train PPO")
	}
	var report struct {
		Valid    bool `json:"provenance_valid"`
		Complete bool `json:"capture_complete"`
		Results  []struct {
			Root      string                 `json:"root"`
			Seed      int                    `json:"seed"`
			Valid     bool                   `json:"capture_valid"`
			Dispatch  bool                   `json:"dispatch_valid"`
			ConfigSHA string                 `json:"provider_config_sha256"`
			Worker    int                    `json:"worker"`
			Mixed     bool                   `json:"fixture_mixed"`
			Goal      *learningenv.GoalStop  `json:"goal_stop"`
			Death     *learningenv.DeathStop `json:"death_stop"`
		} `json:"results"`
	}
	if e = read(filepath.Join(*batch, "report.json"), &report); e != nil {
		return e
	}
	if !report.Valid || !report.Complete {
		return fmt.Errorf("invalid capture/provenance")
	}
	if e = os.Mkdir(*out, 0755); e != nil {
		return e
	}
	rolloutName := "rollout.jsonl"
	if *cudaOnly {
		rolloutName = "native-rollout.jsonl"
	}
	f, e := os.Create(filepath.Join(*out, rolloutName))
	if e != nil {
		return e
	}
	defer f.Close()
	enc := json.NewEncoder(f)
	var contextFile *os.File
	var contextEncoder *json.Encoder
	contextRows := 0
	if behavior.IsRecurrent() {
		contextFile, e = os.Create(filepath.Join(*out, "sequence.jsonl"))
		if e != nil {
			return e
		}
		defer contextFile.Close()
		contextEncoder = json.NewEncoder(contextFile)
	}
	receipts := map[string]string{}
	remember := func(path string) error {
		h, e := sha(path)
		if e == nil {
			receipts[path] = h
		}
		return e
	}
	for _, p := range []string{*model, filepath.Join(*batch, "manifest.json"), filepath.Join(*batch, "report.json")} {
		if e = remember(p); e != nil {
			return e
		}
	}
	exporter := filepath.Join(*batch, "q2combat-export.exe")
	h, e := sha(exporter)
	if e != nil {
		return e
	}
	if !strings.EqualFold(h, manifest.ExporterSHA) {
		return fmt.Errorf("exporter binary changed")
	}
	receipts[exporter] = h
	seen := map[int]bool{}
	count := 0
	skipped := 0
	terminals := 0
	for i, r := range report.Results {
		if !r.Valid || !r.Dispatch || seen[r.Seed] {
			return fmt.Errorf("invalid episode")
		}
		seen[r.Seed] = true
		var cfg struct {
			Test struct {
				SpawnClass  string `json:"spawn_class"`
				TeleportMap string `json:"teleport_map"`
			} `json:"test"`
			Combat struct {
				File string `json:"provider_file"`
			} `json:"combat"`
		}
		if e = read(filepath.Join(r.Root, "bot-config.json"), &cfg); e != nil {
			return e
		}
		h, e := sha(cfg.Combat.File)
		if e != nil {
			return e
		}
		if !strings.EqualFold(h, r.ConfigSHA) {
			return fmt.Errorf("episode model changed")
		}
		p, e := policy.LoadPPO(cfg.Combat.File)
		if e != nil {
			return e
		}
		if p.Version() != behavior.Version() || !p.IsStochastic() || p.SamplingSeed() != int64(r.Seed) {
			return fmt.Errorf("episode weight version or sampling seed differs")
		}
		rewardPath := filepath.Join(r.Root, "dataset", "reward-config.json")
		var rewardConfig learningenv.RewardConfig
		if e = read(rewardPath, &rewardConfig); e != nil {
			return e
		}
		if !reflect.DeepEqual(rewardConfig, manifest.Reward) {
			return fmt.Errorf("reward changed")
		}
		paths := []string{filepath.Join(r.Root, "bot.jsonl"), filepath.Join(r.Root, "server.log"), filepath.Join(r.Root, "reset-expectation.json"), filepath.Join(r.Root, "bot-config.json"), cfg.Combat.File, rewardPath}
		for _, path := range paths {
			if e = remember(path); e != nil {
				return e
			}
		}
		replay := filepath.Join(*out, fmt.Sprintf("replay-%d", i))
		args := []string{"--trace", paths[0], "--server-log", paths[1], "--reset-expectation", paths[2], "--reward-config", rewardPath, "--out", replay, "--worker", fmt.Sprintf("worker-%d", r.Worker), "--episode", fmt.Sprintf("seed-%d", r.Seed), "--end-reason", "game_frame_limit", "--client-name", "SoloRetreatBot", "--require-execution", "--synchronous"}
		if r.Goal != nil {
			if !manifest.StopOnGoal {
				return fmt.Errorf("goal stop not declared by manifest")
			}
			path := filepath.Join(r.Root, "goal-stop.json")
			var goal learningenv.GoalStop
			if e = read(path, &goal); e != nil {
				return e
			}
			if !reflect.DeepEqual(goal, *r.Goal) {
				return fmt.Errorf("goal stop receipt changed")
			}
			if e = remember(path); e != nil {
				return e
			}
			log, err := os.ReadFile(paths[1])
			if err != nil {
				return err
			}
			events, err := learningenv.ReadDamageEvents(strings.NewReader(string(log)))
			if err != nil {
				return err
			}
			native, err := learningenv.ReadNativeSteps(strings.NewReader(string(log)), events)
			if err != nil {
				return err
			}
			trace, err := rows[struct {
				Capture policy.Capture `json:"combat_policy"`
			}](paths[0])
			if err != nil {
				return err
			}
			var observed policy.Observation
			for _, row := range trace {
				if row.Capture.Observation.Identity.Frame == goal.ObservedFrame {
					observed = row.Capture.Observation
				}
			}
			expectedClasses := []string{cfg.Test.SpawnClass}
			if r.Mixed {
				expectedClasses = append(expectedClasses, "monster_gunner")
			}
			if err = learningenv.VerifyGoalStopForClasses(goal, native.Release, events, observed, expectedClasses, cfg.Test.TeleportMap); err != nil {
				return err
			}
			for i := range args {
				if args[i] == "game_frame_limit" {
					args[i] = "combat_goal_complete"
				}
			}
			args = append(args, "--goal-observed-frame", fmt.Sprint(goal.ObservedFrame))
		}
		if r.Death != nil {
			if !manifest.StopOnGoal || r.Goal != nil {
				return fmt.Errorf("death stop requires supervision without a conflicting goal")
			}
			path := filepath.Join(r.Root, "death-stop.json")
			var death learningenv.DeathStop
			if e = read(path, &death); e != nil {
				return e
			}
			if !reflect.DeepEqual(death, *r.Death) {
				return fmt.Errorf("death stop receipt changed")
			}
			if e = remember(path); e != nil {
				return e
			}
			for i := range args {
				if args[i] == "game_frame_limit" {
					args[i] = "combat_first_life_death"
				}
			}
			// The captured exporter verifies the actual ordered native death
			// terminal, receipt observation and full reward before emitting proof.
			args = append(args, "--death-stop", path)
		}
		output, e := exec.Command(exporter, args...).CombinedOutput()
		if e != nil {
			return fmt.Errorf("native replay: %w %s", e, output)
		}
		if r.Death != nil {
			for _, path := range []string{filepath.Join(replay, "death-stop-verification.json"), filepath.Join(r.Root, "dataset", "death-stop-verification.json")} {
				var proof struct {
					State   string                `json:"state"`
					Receipt learningenv.DeathStop `json:"receipt"`
					Step    int                   `json:"death_step"`
				}
				if e = read(path, &proof); e != nil {
					return e
				}
				if proof.State != "verified" || proof.Step <= 0 || !reflect.DeepEqual(proof.Receipt, *r.Death) {
					return fmt.Errorf("death terminal/reward proof missing or changed")
				}
				if e = remember(path); e != nil {
					return e
				}
			}
			a, err := sha(filepath.Join(replay, "death-stop-verification.json"))
			if err != nil {
				return err
			}
			b, err := sha(filepath.Join(r.Root, "dataset", "death-stop-verification.json"))
			if err != nil {
				return err
			}
			if a != b {
				return fmt.Errorf("offline replay differs: death-stop-verification.json")
			}
		}
		for _, name := range []string{"steps.jsonl", "rewards.jsonl", "server_outcomes.jsonl"} {
			a, e := sha(filepath.Join(replay, name))
			if e != nil {
				return e
			}
			b, e := sha(filepath.Join(r.Root, "dataset", name))
			if e != nil {
				return e
			}
			if a != b {
				if name != "rewards.jsonl" {
					return fmt.Errorf("offline replay differs: %s", name)
				}
				if e = verifyRewardReplay(filepath.Join(replay, name), filepath.Join(r.Root, "dataset", name)); e != nil {
					return fmt.Errorf("offline replay differs: %s: %w", name, e)
				}
				if e = remember(filepath.Join(r.Root, "dataset", name)); e != nil {
					return e
				}
			}
			if e = remember(filepath.Join(replay, name)); e != nil {
				return e
			}
		}
		steps, e := rows[learningenv.Step](filepath.Join(replay, "steps.jsonl"))
		if e != nil {
			return e
		}
		logFile, e := os.Open(filepath.Join(r.Root, "server.log"))
		if e != nil {
			return e
		}
		target, proofErr := learningenv.VerifyCurriculum(logFile, manifest.TrainingHealth, r.Seed)
		logFile.Close()
		if proofErr != nil {
			return proofErr
		}
		if manifest.PostFrameRNG {
			logFile, e = os.Open(filepath.Join(r.Root, "server.log"))
			if e != nil {
				return e
			}
			weapon := "Blaster"
			if manifest.Loadout == "machinegun" || manifest.Loadout == "weapons" || manifest.Loadout == "weapons-scarce" {
				weapon = "Machinegun"
			}
			proofErr = learningenv.VerifyPostFrameRNG(logFile, r.Seed, manifest.ReleaseFrame, weapon)
			logFile.Close()
			if proofErr != nil {
				return proofErr
			}
			if manifest.ReleaseFrame > 0 {
				idleFrame := 9
				if weapon == "Machinegun" {
					idleFrame = 6
				}
				for _, step := range steps {
					if step.Owner == "provider" {
						if step.Observation.GunFrame != idleFrame {
							return fmt.Errorf("first policy weapon phase differs")
						}
						break
					}
				}
			}
		}
		if target != 0 {
			found := false
			if len(steps) > 0 {
				for _, enemy := range steps[0].Observation.Enemies {
					if enemy.ID == target && enemy.Class == "monster_parasite" {
						found = true
					}
				}
			}
			if !found {
				return fmt.Errorf("curriculum target not in initial observation")
			}
		}
		rewards, e := rows[learningenv.Reward](filepath.Join(replay, "rewards.jsonl"))
		if e != nil {
			return e
		}
		if len(steps) != len(rewards) {
			return fmt.Errorf("reward length")
		}
		for j, s := range steps {
			reward := rewards[j]
			if s.Observation.Identity.Life == 1 && s.Owner == "provider" && (s.Provider != p.Version() || s.Sample == nil) {
				return fmt.Errorf("missing or foreign on-policy sample")
			}
			if s.Observation.Identity.Life == 1 && s.Owner == "provider" && s.Provider == p.Version() && s.Sample != nil {
				if !*cudaOnly {
					e = p.VerifyMemory(s.Observation, *s.Sample)
				}
				if e != nil {
					return e
				}
				if contextEncoder != nil {
					x, err := policy.FeaturesForVersion(s.Observation, p.FeatureVersion())
					if err != nil {
						return err
					}
					if err = contextEncoder.Encode(map[string]any{"features": x, "seed": r.Seed, "index": s.Index, "frame": s.Observation.Identity.Frame, "memory": s.Sample.Memory}); err != nil {
						return err
					}
					contextRows++
				}
			}
			if s.Observation.Identity.Life != 1 || s.Owner != "provider" || s.Provider != p.Version() || s.Sample == nil || s.Next == nil || !reward.Available || reward.Score == nil {
				skipped++
				continue
			}
			if reward.Step != s.Index || s.Sample.SamplingSeed != int64(r.Seed) {
				return fmt.Errorf("sample/reward identity")
			}
			a, e := p.ActionForSample(s.Observation, *s.Sample)
			if e != nil {
				return e
			}
			if !reflect.DeepEqual(a, s.Action) {
				return fmt.Errorf("behavior sample altered")
			}
			if !*cudaOnly {
				_, lp, value, err := p.Review(s.Observation, *s.Sample)
				if err != nil {
					return err
				}
				if math.Abs(lp-s.Sample.LogProbability) > 1e-8 || math.Abs(value-s.Sample.Value) > 1e-8 {
					return fmt.Errorf("behavior probability/value altered")
				}
			}
			features, e := policy.FeaturesForVersion(s.Observation, p.FeatureVersion())
			if e != nil {
				return e
			}
			nv := 0.0
			zero := s.Terminal || manifest.Reward.HasKillReward() && s.Truncated && s.Reason == "control_handoff"
			if !*cudaOnly && !zero {
				nv, e = p.ValueAfter(s.Observation, *s.Sample, *s.Next)
				if e != nil {
					return e
				}
			}
			if s.Terminal {
				terminals++
			}
			exported := row{Features: features, NextValue: nv, Sample: *s.Sample, Reward: *reward.Score, Terminal: s.Terminal, Truncated: s.Truncated, Seed: r.Seed, Index: s.Index, Frame: s.Observation.Identity.Frame, NextFrame: s.Next.Identity.Frame, Interventions: s.Interventions}
			if *cudaOnly {
				exported.NextFeatures, e = policy.FeaturesForVersion(*s.Next, p.FeatureVersion())
				if e != nil {
					return e
				}
				exported.NextReset = !policy.SameLife(s.Observation.Identity, s.Next.Identity) || s.Next.Identity.Frame != s.Observation.Identity.Frame+1
				exported.BootstrapZero = zero
			}
			if e = enc.Encode(exported); e != nil {
				return e
			}
			count++
		}
	}
	if count == 0 {
		return fmt.Errorf("no on-policy transitions")
	}
	for path, h := range receipts {
		now, e := sha(path)
		if e != nil || now != h {
			return fmt.Errorf("input changed: %s", path)
		}
	}
	if e = f.Sync(); e != nil {
		return e
	}
	rolloutSHA, e := sha(filepath.Join(*out, rolloutName))
	if e != nil {
		return e
	}
	metadata := map[string]any{"version": "combat_ppo_rollout_v1", "feature_version": behavior.FeatureVersion(), "reward_config_sha256": strings.ToLower(manifest.RewardSHA), "reward_version": manifest.Reward.Version, "aim_gamma": manifest.Reward.AimGamma, "training_monster_health": manifest.TrainingHealth, "rows": count, "skipped": skipped, "terminals": terminals, "policy_version": behavior.Version(), "model_sha256": modelSHA, "rollout_sha256": rolloutSHA, "source_sha256": receipts, "scope": "Fresh stochastic provider transitions, first life only; guards retained as environment execution; gaps cut in GAE; v2 verified control handoff retains reward with zero segment bootstrap; full world reset equivalence unproven"}
	if *cudaOnly {
		metadata["version"] = "combat_ppo_native_cuda_pending_v1"
		metadata["numerical_verification"] = "cuda_pending"
		metadata["native_rollout_sha256"] = rolloutSHA
		delete(metadata, "rollout_sha256")
	}
	if contextFile != nil {
		if e = contextFile.Sync(); e != nil {
			return e
		}
		h, err := sha(filepath.Join(*out, "sequence.jsonl"))
		if err != nil {
			return err
		}
		metadata["recurrent_version"] = behavior.MemoryVersion()
		metadata["sequence_sha256"] = h
		metadata["sequence_rows"] = contextRows
	}
	data, _ := json.MarshalIndent(metadata, "", "  ")
	fmt.Printf("PPO native export: rows=%d terminal=%d policy=%s CUDA_pending=%v\n", count, terminals, behavior.Version(), *cudaOnly)
	return os.WriteFile(filepath.Join(*out, "report.json"), data, 0644)
}
