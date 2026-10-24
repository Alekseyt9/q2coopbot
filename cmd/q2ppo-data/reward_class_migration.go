package main

import (
	"fmt"
	"math"
	"path/filepath"
	"q2coopbot/internal/learningenv"
	"reflect"
)

func commitmentStateDeltaOnly(a, b learningenv.Reward) bool {
	if !a.Available || !b.Available || a.Score == nil || b.Score == nil {
		return false
	}
	x, y := *a.Score, *b.Score
	delta := a.Components["target_commitment"] - b.Components["target_commitment"]
	copyComponents := func(values map[string]float64) map[string]float64 {
		result := map[string]float64{}
		for key, value := range values {
			if key != "target_commitment" {
				result[key] = value
			}
		}
		return result
	}
	a.Components, b.Components = copyComponents(a.Components), copyComponents(b.Components)
	a.Score, b.Score, a.TargetCommitment, b.TargetCommitment = nil, nil, nil, nil
	return reflect.DeepEqual(a, b) && !math.IsNaN(x) && !math.IsNaN(y) &&
		!math.IsInf(x, 0) && !math.IsInf(y, 0) && math.Abs((x-y)-delta) <= 1e-10
}

// Only the explicit corrected-exporter route admits class reconciliation.
// Recompute every scalar reward independently from byte-identical steps and
// native effects. Nothing here evaluates a neural model or changes its inputs.
func verifyCanonicalClassRewardReplay(replay, captured string, config learningenv.RewardConfig) (int, error) {
	if config.Version != learningenv.BlockedGroundRequestRewardVersion {
		return 0, fmt.Errorf("class reconciliation requires pinned v14 objective")
	}
	for _, name := range []string{"steps.jsonl", "server_outcomes.jsonl"} {
		a, err := sha(filepath.Join(replay, name))
		if err != nil {
			return 0, err
		}
		b, err := sha(filepath.Join(captured, name))
		if err != nil || a != b {
			return 0, fmt.Errorf("class reconciliation changed %s", name)
		}
	}
	steps, err := rows[learningenv.Step](filepath.Join(replay, "steps.jsonl"))
	if err != nil {
		return 0, err
	}
	effects, err := rows[learningenv.ServerOutcome](filepath.Join(replay, "server_outcomes.jsonl"))
	if err != nil {
		return 0, err
	}
	actual, err := rows[learningenv.Reward](filepath.Join(replay, "rewards.jsonl"))
	if err != nil {
		return 0, err
	}
	before, err := rows[learningenv.Reward](filepath.Join(captured, "rewards.jsonl"))
	if err != nil {
		return 0, err
	}
	if len(steps) != len(effects) || len(steps) != len(actual) || len(actual) != len(before) {
		return 0, fmt.Errorf("class reconciliation row count differs")
	}
	sequence := learningenv.SequenceReward{Config: config}
	repaired := 0
	for i := range steps {
		s, effect, a, b := &steps[i], effects[i], actual[i], before[i]
		if s.Index != effect.Step || s.Worker != effect.Worker || s.Episode != effect.Episode ||
			a.Step != b.Step || a.Version != b.Version || a.Worker != b.Worker || a.Episode != b.Episode {
			return 0, fmt.Errorf("class reconciliation identity changed at %d", i)
		}
		expected := sequence.Evaluate(s, effect)
		if !equivalentRewardReplay(expected, a) {
			return 0, fmt.Errorf("independent scalar reward differs at %d", i)
		}
		if equivalentRewardReplay(a, b) {
			continue
		}
		if !b.Available && b.Reason == "commitment_target_class_mismatch" && a.Available {
			repaired++
			continue
		}
		if repaired == 0 || !commitmentStateDeltaOnly(a, b) {
			return 0, fmt.Errorf("unrelated reward change at %d", i)
		}
	}
	return repaired, nil
}
