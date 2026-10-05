package demodata

import (
	"bufio"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"

	"q2coopbot/internal/harness"
	"q2coopbot/internal/learningenv"
	"q2coopbot/internal/policy"
	"q2coopbot/internal/quake"
)

type Condition struct {
	Map         string `json:"map"`
	Loadout     string `json:"loadout"`
	Mixed       bool   `json:"mixed"`
	HealthKit   bool   `json:"health_kit"`
	Synchronous bool   `json:"synchronous"`
}
type EpisodeSpec struct {
	Seed  int    `json:"seed"`
	Split string `json:"split"`
}
type Spec struct {
	Version          string        `json:"version"`
	SelectionVersion string        `json:"selection_version"`
	Condition        Condition     `json:"condition"`
	Episodes         []EpisodeSpec `json:"episodes"`
}

func (s Spec) Validate() error {
	if s.Version != "combat_dataset_spec_v1" || s.SelectionVersion != SelectionVersion || s.Condition.Map == "" || !s.Condition.Synchronous || len(s.Episodes) == 0 {
		return fmt.Errorf("invalid dataset specification")
	}
	seen := map[int]bool{}
	splits := map[string]int{}
	for _, e := range s.Episodes {
		if e.Seed < 0 || int64(e.Seed) > 2147483647 || seen[e.Seed] || (e.Split != "train" && e.Split != "validation" && e.Split != "test") {
			return fmt.Errorf("duplicate seed or invalid split")
		}
		seen[e.Seed] = true
		splits[e.Split]++
	}
	for _, name := range []string{"train", "validation", "test"} {
		if splits[name] == 0 {
			return fmt.Errorf("missing split %s", name)
		}
	}
	return nil
}

type Source struct {
	Batch      string            `json:"batch"`
	Assistance string            `json:"teacher_assistance"`
	Seed       int               `json:"seed"`
	Split      string            `json:"split"`
	Controller string            `json:"controller_source_fingerprint"`
	Native     string            `json:"native_source_fingerprint"`
	Files      map[string]string `json:"sha256"`
}
type Candidate struct {
	Version     string             `json:"version"`
	Source      Source             `json:"source"`
	Step        int                `json:"step"`
	Worker      string             `json:"worker"`
	Episode     string             `json:"episode"`
	Selection   Selection          `json:"selection"`
	Observation policy.Observation `json:"observation"`
	Target      policy.Action      `json:"target_applied_action"`
}
type Counts struct {
	Episodes   int            `json:"episodes"`
	Steps      int            `json:"steps"`
	Candidates int            `json:"candidates"`
	Movement   int            `json:"movement"`
	AimAttack  int            `json:"aim_attack"`
	Reasons    map[string]int `json:"rejections"`
}
type Report struct {
	Version string             `json:"version"`
	Ready   bool               `json:"candidate_dataset_ready"`
	SpecSHA string             `json:"spec_sha256"`
	Splits  map[string]*Counts `json:"splits"`
	Sources []Source           `json:"sources"`
	Scope   string             `json:"scope"`
}

type batchManifest struct {
	Provenance  bool                      `json:"provenance_valid"`
	Source      string                    `json:"source_fingerprint"`
	Native      string                    `json:"native_source_fingerprint"`
	Mode        string                    `json:"provider"`
	Loadout     string                    `json:"loadout"`
	Mixed       bool                      `json:"mixed"`
	HealthKit   bool                      `json:"health_kit"`
	Synchronous bool                      `json:"synchronous"`
	Reward      *learningenv.RewardConfig `json:"reward_config"`
}
type episodeResult struct {
	Seed          int    `json:"seed"`
	Worker        int    `json:"worker"`
	Episode       int    `json:"episode"`
	Root          string `json:"root"`
	Valid         bool   `json:"capture_valid"`
	SeedConfirmed bool   `json:"seed_confirmed"`
	RuntimeFiles  []struct {
		Path string `json:"path"`
		SHA  string `json:"sha256"`
	} `json:"runtime_files"`
}
type batchReport struct {
	Complete   bool            `json:"capture_complete"`
	Provenance bool            `json:"provenance_valid"`
	Results    []episodeResult `json:"results"`
}
type inputEpisode struct {
	Batch    string
	Manifest batchManifest
	Result   episodeResult
}

func readJSON(path string, value any) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	return json.Unmarshal(data, value)
}
func fileSHA(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	hash := sha256.New()
	if _, err := io.Copy(hash, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(hash.Sum(nil)), nil
}
func readLines[T any](path string) ([]T, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 65536), 8*1024*1024)
	var result []T
	for scanner.Scan() {
		var value T
		if err := json.Unmarshal(scanner.Bytes(), &value); err != nil {
			return nil, err
		}
		result = append(result, value)
	}
	return result, scanner.Err()
}

func Build(specPath string, batches []string, out string) (Report, error) {
	report := Report{Version: DatasetVersion, Splits: map[string]*Counts{}, Scope: "Automatically selected component candidates, not human-reviewed demonstrations or trained policy. Seed-disjoint splits share one fixture; no unseen-map acceptance."}
	data, err := os.ReadFile(specPath)
	if err != nil {
		return report, err
	}
	var spec Spec
	if err := json.Unmarshal(data, &spec); err != nil {
		return report, err
	}
	if err := spec.Validate(); err != nil {
		return report, err
	}
	hash := sha256.Sum256(data)
	report.SpecSHA = hex.EncodeToString(hash[:])
	inputs := map[int]inputEpisode{}
	for _, batch := range batches {
		batch, err = filepath.Abs(batch)
		if err != nil {
			return report, err
		}
		var manifest batchManifest
		var summary batchReport
		if err := readJSON(filepath.Join(batch, "manifest.json"), &manifest); err != nil {
			return report, err
		}
		if err := readJSON(filepath.Join(batch, "report.json"), &summary); err != nil {
			return report, err
		}
		if !manifest.Provenance || !summary.Provenance || !summary.Complete || manifest.Source == "" || manifest.Native == "" || manifest.Reward == nil {
			return report, fmt.Errorf("unverified batch or missing reward: %s", batch)
		}
		if err := manifest.Reward.Validate(); err != nil {
			return report, err
		}
		if manifest.Loadout != spec.Condition.Loadout || manifest.Mixed != spec.Condition.Mixed || manifest.HealthKit != spec.Condition.HealthKit || manifest.Synchronous != spec.Condition.Synchronous {
			return report, fmt.Errorf("batch does not match frozen condition")
		}
		for _, result := range summary.Results {
			if !result.Valid || !result.SeedConfirmed {
				return report, fmt.Errorf("invalid source episode")
			}
			rel, err := filepath.Rel(batch, result.Root)
			if err != nil || rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) || filepath.IsAbs(rel) {
				return report, fmt.Errorf("episode outside batch")
			}
			if _, exists := inputs[result.Seed]; exists {
				return report, fmt.Errorf("reused seed %d", result.Seed)
			}
			inputs[result.Seed] = inputEpisode{batch, manifest, result}
		}
	}
	if len(inputs) != len(spec.Episodes) {
		return report, fmt.Errorf("source episodes differ from frozen seed assignments")
	}
	for _, e := range spec.Episodes {
		if _, ok := inputs[e.Seed]; !ok {
			return report, fmt.Errorf("missing planned seed %d", e.Seed)
		}
	}
	if err := os.Mkdir(out, 0755); err != nil {
		return report, err
	}
	if err := os.WriteFile(filepath.Join(out, "spec.json"), data, 0644); err != nil {
		return report, err
	}
	encoders := map[string]*json.Encoder{}
	for _, name := range []string{"train", "validation", "test", "diagnostics"} {
		f, err := os.Create(filepath.Join(out, name+".jsonl"))
		if err != nil {
			return report, err
		}
		defer f.Close()
		encoders[name] = json.NewEncoder(f)
		if name != "diagnostics" {
			report.Splits[name] = &Counts{Reasons: map[string]int{}}
		}
	}
	for _, assignment := range spec.Episodes {
		input := inputs[assignment.Seed]
		root := input.Result.Root
		source := Source{Batch: input.Batch, Seed: assignment.Seed, Split: assignment.Split, Controller: input.Manifest.Source, Native: input.Manifest.Native, Files: map[string]string{}}
		source.Assistance = "rules tactical controller and fixture placement"
		if input.Manifest.Loadout == "shotgun" {
			source.Assistance += "; test-only fixed Shotgun, automatic weapon selection disabled"
		}
		paths := map[string]string{"manifest": filepath.Join(input.Batch, "manifest.json"), "report": filepath.Join(input.Batch, "report.json"), "config": filepath.Join(root, "bot-config.json"), "trace": filepath.Join(root, "bot.jsonl"), "server": filepath.Join(root, "server.log"), "steps": filepath.Join(root, "dataset/steps.jsonl"), "effects": filepath.Join(root, "dataset/server_outcomes.jsonl"), "rewards": filepath.Join(root, "dataset/rewards.jsonl"), "initial": filepath.Join(root, "dataset/episode_start.json"), "reset": filepath.Join(root, "reset-expectation.json")}
		if len(input.Result.RuntimeFiles) == 0 {
			return report, fmt.Errorf("native runtime hashes missing")
		}
		for _, record := range input.Result.RuntimeFiles {
			path := filepath.Clean(filepath.FromSlash(record.Path))
			if filepath.IsAbs(path) || path == ".." || strings.HasPrefix(path, ".."+string(filepath.Separator)) {
				return report, fmt.Errorf("runtime file outside episode")
			}
			full := filepath.Join(root, "runtime", path)
			value, err := fileSHA(full)
			if err != nil {
				return report, err
			}
			if !strings.EqualFold(value, record.SHA) {
				return report, fmt.Errorf("runtime file hash mismatch")
			}
			paths["runtime:"+record.Path] = full
		}
		for name, path := range paths {
			value, err := fileSHA(path)
			if err != nil {
				return report, err
			}
			source.Files[name] = value
		}
		count := report.Splits[assignment.Split]
		count.Episodes++
		err := verifyEpisode(input, spec.Condition.Map, func(s learningenv.Step, effects learningenv.ServerOutcome, reward learningenv.Reward, c policy.Capture) error {
			selection := Select(s, effects, c, input.Manifest.Mode)
			count.Steps++
			if selection.Quality != "rejected" {
				candidate := Candidate{DatasetVersion, source, s.Index, s.Worker, s.Episode, selection, s.Observation, s.AppliedAction}
				if err := encoders[assignment.Split].Encode(candidate); err != nil {
					return err
				}
				count.Candidates++
				if selection.Heads.Movement {
					count.Movement++
				}
				if selection.Heads.Aim {
					count.AimAttack++
				}
			} else {
				count.Reasons[selection.Reason]++
			}
			return encoders["diagnostics"].Encode(struct {
				Source    Source                    `json:"source"`
				Selection Selection                 `json:"selection"`
				Step      learningenv.Step          `json:"step"`
				Effects   learningenv.ServerOutcome `json:"effects"`
				Reward    learningenv.Reward        `json:"reward"`
			}{source, selection, s, effects, reward})
		})
		if err != nil {
			return report, fmt.Errorf("seed %d: %w", assignment.Seed, err)
		}
		for name, path := range paths {
			value, err := fileSHA(path)
			if err != nil {
				return report, err
			}
			if value != source.Files[name] {
				return report, fmt.Errorf("source changed during dataset build")
			}
		}
		report.Sources = append(report.Sources, source)
	}
	current, err := fileSHA(specPath)
	if err != nil {
		return report, err
	}
	if current != report.SpecSHA {
		return report, fmt.Errorf("spec changed during build")
	}
	report.Ready = true
	for _, name := range []string{"train", "validation", "test"} {
		if report.Splits[name].Candidates == 0 {
			report.Ready = false
		}
	}
	sort.Slice(report.Sources, func(i, j int) bool { return report.Sources[i].Seed < report.Sources[j].Seed })
	output, err := json.MarshalIndent(report, "", "  ")
	if err != nil {
		return report, err
	}
	if err := os.WriteFile(filepath.Join(out, "report.json"), output, 0644); err != nil {
		return report, err
	}
	if !report.Ready {
		return report, fmt.Errorf("empty candidate split; retain diagnostics, do not train")
	}
	return report, nil
}

func verifyEpisode(input inputEpisode, mapName string, emit func(learningenv.Step, learningenv.ServerOutcome, learningenv.Reward, policy.Capture) error) error {
	root := input.Result.Root
	steps, err := readLines[learningenv.Step](filepath.Join(root, "dataset/steps.jsonl"))
	if err != nil {
		return err
	}
	effects, err := readLines[learningenv.ServerOutcome](filepath.Join(root, "dataset/server_outcomes.jsonl"))
	if err != nil {
		return err
	}
	rewards, err := readLines[learningenv.Reward](filepath.Join(root, "dataset/rewards.jsonl"))
	if err != nil {
		return err
	}
	if len(steps) == 0 || len(steps) != len(effects) || len(steps) != len(rewards) {
		return fmt.Errorf("unaligned dataset files")
	}
	log, err := os.Open(filepath.Join(root, "server.log"))
	if err != nil {
		return err
	}
	events, err := learningenv.ReadDamageEvents(log)
	log.Close()
	if err != nil {
		return err
	}
	log, err = os.Open(filepath.Join(root, "server.log"))
	if err != nil {
		return err
	}
	native, err := learningenv.ReadNativeSteps(log, events)
	log.Close()
	if err != nil {
		return err
	}
	if native.Release.Seed != input.Result.Seed {
		return fmt.Errorf("native seed mismatch")
	}
	applied, err := harness.ReadAppliedCommandsForClient(filepath.Join(root, "server.log"), "SoloRetreatBot")
	if err != nil {
		return err
	}
	executions := learningenv.NewExecutionIndex(applied)
	joiner := learningenv.DamageJoiner{Events: events}
	var expected learningenv.ResetExpectation
	if err := readJSON(filepath.Join(root, "reset-expectation.json"), &expected); err != nil {
		return err
	}
	if expected.Seed == nil || *expected.Seed != input.Result.Seed {
		return fmt.Errorf("fixture seed mismatch")
	}
	var config struct {
		Combat struct {
			Mode string `json:"mode"`
		} `json:"combat"`
		Test struct {
			Synchronous bool   `json:"synchronous"`
			Fixture     string `json:"weapon_switch_fixture"`
		} `json:"test"`
	}
	if err := readJSON(filepath.Join(root, "bot-config.json"), &config); err != nil {
		return err
	}
	if config.Combat.Mode != input.Manifest.Mode || config.Test.Synchronous != input.Manifest.Synchronous || config.Test.Fixture != "parasite_"+input.Manifest.Loadout {
		return fmt.Errorf("teacher configuration differs from manifest")
	}
	switch input.Manifest.Loadout {
	case "blaster":
		if expected.Weapon != "Blaster" || expected.Ammo != 0 {
			return fmt.Errorf("Blaster reset differs from loadout")
		}
	case "shotgun":
		if expected.Weapon != "Shotgun" || expected.Ammo != 20 {
			return fmt.Errorf("Shotgun reset differs from loadout")
		}
	default:
		return fmt.Errorf("unsupported synchronous teacher loadout")
	}
	var initial struct {
		Version     string                  `json:"version"`
		Observation *policy.Observation     `json:"first_usable_observation"`
		Proof       *learningenv.ResetProof `json:"observed_reset_proof"`
	}
	if err := readJSON(filepath.Join(root, "dataset/episode_start.json"), &initial); err != nil {
		return err
	}
	rows, err := readLines[struct {
		Capture  *policy.Capture `json:"combat_policy"`
		Command  quake.UserCmd   `json:"sent_command"`
		Sequence uint32          `json:"client_sequence"`
	}](filepath.Join(root, "bot.jsonl"))
	if err != nil {
		return err
	}
	assembler := learningenv.Assembler{Worker: fmt.Sprintf("worker-%d", input.Result.Worker), Episode: fmt.Sprintf("seed-%d", input.Result.Seed)}
	count := 0
	captures := map[uint32]policy.Capture{}
	var sent []harness.Trace
	verify := func(s *learningenv.Step) error {
		if s == nil {
			return nil
		}
		if count >= len(steps) || s.Observation.Identity.Map != mapName {
			return fmt.Errorf("step/map count mismatch")
		}
		if count == 0 {
			proof := learningenv.VerifyReset(s.Observation, expected)
			if err := proof.Error(); err != nil {
				return err
			}
			if s.Observation.Identity.Frame != native.Release.Frame || s.Observation.Identity.Spawncount != native.Release.Spawncount || s.Observation.Inventory == nil || s.Observation.InventoryAgeFrames == nil || *s.Observation.InventoryAgeFrames < 0 || *s.Observation.InventoryAgeFrames > 2 {
				return fmt.Errorf("initial release/inventory mismatch")
			}
			proof.NativeBarrier = native.Release
			if initial.Version != "observed_episode_start_v1" || initial.Observation == nil || !reflect.DeepEqual(*initial.Observation, s.Observation) || !reflect.DeepEqual(initial.Proof, &proof) {
				return fmt.Errorf("exported reset proof differs from source")
			}
		}
		x := executions.Match(s)
		s.Execution = &x
		n, err := native.Match(s)
		if err != nil {
			return err
		}
		s.Native = n
		if !x.Matched || s.Next != nil && !x.WindowExclusive {
			return fmt.Errorf("unproven exact dispatch")
		}
		o := joiner.JoinNative(s, n)
		r := input.Manifest.Reward.Evaluate(s, o)
		if !reflect.DeepEqual(*s, steps[count]) || !reflect.DeepEqual(o, effects[count]) || !reflect.DeepEqual(r, rewards[count]) {
			return fmt.Errorf("export differs from reassembled native/observed source at step %d", s.Index)
		}
		c := captures[s.ClientSequence]
		count++
		return emit(*s, o, r, c)
	}
	for _, row := range rows {
		if row.Capture == nil || row.Capture.AppliedCommand != row.Command || row.Capture.ClientSequence != row.Sequence {
			return fmt.Errorf("invalid capture/sent command")
		}
		if _, ok := captures[row.Sequence]; ok {
			return fmt.Errorf("reused capture sequence")
		}
		captures[row.Sequence] = *row.Capture
		id := row.Capture.Observation.Identity
		sent = append(sent, harness.Trace{Connection: id.Connection, Generation: id.Spawncount, ClientSequence: row.Sequence, Command: row.Command})
		s, _, err := assembler.Push(*row.Capture)
		if err != nil {
			return err
		}
		if err := verify(s); err != nil {
			return err
		}
	}
	s, _ := assembler.Close("game_frame_limit")
	if err := verify(s); err != nil {
		return err
	}
	proof := harness.VerifyAppliedCommands(sent, applied)
	if !proof.Accepted || len(native.Steps) != len(rows) || count != len(steps) {
		return fmt.Errorf("full command/capture count proof failed")
	}
	return nil
}
