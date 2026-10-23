package policy

import (
	"fmt"
	"math"
	"math/rand"
)

const ExecutionGreedyTargetMode = "greedy_target_mode_v1"
const ExecutionGreedyTargetModeFire = "greedy_target_mode_fire_v1"

func validateExecutionMode(f PPOFile) error {
	if f.GeometryAimExplore != nil && f.ExecutionMode != ExecutionGeometryAimExplore {
		return fmt.Errorf("exploration requires its predictive execution mode")
	}
	if f.ExecutionMode == ExecutionGeometryAimExplore {
		if f.Deterministic || f.TargetHead != TargetHeadVersion || f.AimModeHead != PrecisionHeadVersion {
			return fmt.Errorf("invalid predictive exploration mode")
		}
		return validateGeometryAimExplore(f)
	}
	if f.ExecutionMode == ExecutionGeometryAimRanker {
		if f.Deterministic || f.TargetHead != TargetHeadVersion || f.AimModeHead != PrecisionHeadVersion {
			return fmt.Errorf("invalid inference-only geometry ranker")
		}
		return validateGeometryAimRanker(f)
	}
	if f.GeometryAimRanker != nil {
		return fmt.Errorf("geometry ranker requires its inference execution mode")
	}
	if f.ExecutionMode == "" {
		return nil
	}
	if (f.ExecutionMode != ExecutionGreedyTargetMode && f.ExecutionMode != ExecutionGreedyTargetModeFire) ||
		f.Deterministic || f.TargetHead != TargetHeadVersion || f.AimModeHead != PrecisionHeadVersion {
		return fmt.Errorf("invalid inference-only execution mode")
	}
	return nil
}

func greedyExecutionCategories(mode string) bool {
	return mode == ExecutionGreedyTargetMode || mode == ExecutionGreedyTargetModeFire
}

// Greedy head choices still consume their legacy uniform. This keeps movement,
// fire, vertical and weapon randomness coupled across paired execution modes.
func selectExecutionTarget(probs []float64, rng *rand.Rand, deterministic, greedy bool) int {
	u := 0.0
	if !deterministic {
		u = rng.Float64()
	}
	selected := 0
	if deterministic || greedy {
		for i, probability := range probs {
			if probability > probs[selected] {
				selected = i
			}
		}
		return selected
	}
	for i, probability := range probs {
		if math.IsInf(probability, -1) {
			continue
		}
		selected = i
		u -= math.Exp(probability)
		if u <= 0 {
			break
		}
	}
	return selected
}

func selectExecutionAimMode(probs []float64, rng *rand.Rand, deterministic, greedy bool) int {
	u := 0.0
	if !deterministic {
		u = rng.Float64()
	}
	if deterministic || greedy {
		if probs[1] > probs[0] {
			return 1
		}
		return 0
	}
	if u >= math.Exp(probs[0]) {
		return 1
	}
	return 0
}
