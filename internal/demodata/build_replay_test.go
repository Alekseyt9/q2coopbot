package demodata

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"q2coopbot/internal/learningenv"
	"q2coopbot/internal/policy"
)

// Opt-in integration test replays an existing immutable harness batch. Only
// temporary copies are changed; ordinary unit tests need no game assets.
func TestNativeReplayRejectsChangedExports(t *testing.T) {
	batch := os.Getenv("Q2_DEMONSTRATION_BATCH")
	if batch == "" {
		t.Skip("set Q2_DEMONSTRATION_BATCH to a completed synchronous rules batch")
	}
	var manifest batchManifest
	var summary batchReport
	if err := readJSON(filepath.Join(batch, "manifest.json"), &manifest); err != nil {
		t.Fatal(err)
	}
	if err := readJSON(filepath.Join(batch, "report.json"), &summary); err != nil {
		t.Fatal(err)
	}
	if !summary.Complete || len(summary.Results) == 0 {
		t.Fatal("incomplete replay fixture")
	}
	input := inputEpisode{Batch: batch, Manifest: manifest, Result: summary.Results[0]}
	emitted := 0
	emit := func(learningenv.Step, learningenv.ServerOutcome, learningenv.Reward, policy.Capture) error {
		emitted++
		return nil
	}
	if err := verifyEpisode(input, "base1", emit); err != nil || emitted == 0 {
		t.Fatalf("original native replay: %v; emitted=%d", err, emitted)
	}
	for _, name := range []string{"command_label", "reward", "reset_seed", "teacher_config"} {
		t.Run(name, func(t *testing.T) {
			copyInput := input
			copyInput.Result.Root = t.TempDir()
			for _, path := range []string{"server.log", "bot.jsonl", "bot-config.json", "reset-expectation.json", "dataset/steps.jsonl", "dataset/server_outcomes.jsonl", "dataset/rewards.jsonl", "dataset/episode_start.json"} {
				data, err := os.ReadFile(filepath.Join(input.Result.Root, path))
				if err != nil {
					t.Fatal(err)
				}
				out := filepath.Join(copyInput.Result.Root, path)
				if err := os.MkdirAll(filepath.Dir(out), 0755); err != nil {
					t.Fatal(err)
				}
				if err := os.WriteFile(out, data, 0644); err != nil {
					t.Fatal(err)
				}
			}
			want := "export differs from reassembled"
			switch name {
			case "teacher_config":
				path := filepath.Join(copyInput.Result.Root, "bot-config.json")
				var config map[string]any
				if err := readJSON(path, &config); err != nil {
					t.Fatal(err)
				}
				config["test"].(map[string]any)["weapon_switch_fixture"] = "parasite_other"
				data, err := json.Marshal(config)
				if err != nil {
					t.Fatal(err)
				}
				if err := os.WriteFile(path, data, 0644); err != nil {
					t.Fatal(err)
				}
				want = "teacher configuration differs"
			case "command_label":
				path := filepath.Join(copyInput.Result.Root, "dataset/steps.jsonl")
				rows, err := readLines[learningenv.Step](path)
				if err != nil {
					t.Fatal(err)
				}
				rows[0].Command.Forward++
				writeReplayRows(t, path, rows)
			case "reward":
				path := filepath.Join(copyInput.Result.Root, "dataset/rewards.jsonl")
				rows, err := readLines[learningenv.Reward](path)
				if err != nil {
					t.Fatal(err)
				}
				value := 12345.0
				rows[0].Score = &value
				writeReplayRows(t, path, rows)
			case "reset_seed":
				path := filepath.Join(copyInput.Result.Root, "reset-expectation.json")
				var expected learningenv.ResetExpectation
				if err := readJSON(path, &expected); err != nil {
					t.Fatal(err)
				}
				seed := input.Result.Seed + 1
				expected.Seed = &seed
				data, err := json.Marshal(expected)
				if err != nil {
					t.Fatal(err)
				}
				if err := os.WriteFile(path, data, 0644); err != nil {
					t.Fatal(err)
				}
				want = "fixture seed mismatch"
			}
			if err := verifyEpisode(copyInput, "base1", emit); err == nil || !strings.Contains(err.Error(), want) {
				t.Fatalf("changed %s accepted or unexpected failure: %v", name, err)
			}
		})
	}
}

func writeReplayRows[T any](t *testing.T, path string, rows []T) {
	t.Helper()
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	encoder := json.NewEncoder(f)
	for _, row := range rows {
		if err := encoder.Encode(row); err != nil {
			t.Fatal(err)
		}
	}
}
