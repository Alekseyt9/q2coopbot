package policy

import (
	"math"
	"math/rand"
	"testing"
)

// These tests exercise categorical samplers and metadata only, without NN inference.
func TestExecutionModeContractAndRandomStream(t *testing.T) {
	f := PPOFile{TargetHead: TargetHeadVersion, AimModeHead: PrecisionHeadVersion}
	if err := validateExecutionMode(f); err != nil {
		t.Fatal(err)
	}
	if !(&PPO{file: f}).IsStochastic() {
		t.Fatal("legacy stochastic policy must remain eligible for standard PPO")
	}
	for _, mode := range []string{ExecutionGreedyTargetMode, ExecutionGreedyTargetModeFire} {
		f.ExecutionMode = mode
		if err := validateExecutionMode(f); err != nil {
			t.Fatal(err)
		}
		if (&PPO{file: f}).IsStochastic() {
			t.Fatal("partially greedy execution must not export standard PPO data")
		}
		f.Deterministic = true
		if validateExecutionMode(f) == nil {
			t.Fatal("mixed and global deterministic flags must not coexist")
		}
		f.Deterministic = false
	}
	f.ExecutionMode = "unknown"
	if validateExecutionMode(f) == nil {
		t.Fatal("unknown execution mode accepted")
	}
	f.ExecutionMode, f.TargetHead = ExecutionGreedyTargetMode, ""
	if validateExecutionMode(f) == nil {
		t.Fatal("missing categorical target head accepted")
	}

	probs := []float64{math.Log(.1), math.Inf(-1), math.Log(.9)}
	aim := []float64{math.Log(.2), math.Log(.8)}
	legacy, greedy := rand.New(rand.NewSource(123)), rand.New(rand.NewSource(123))
	for i := 0; i < 1000; i++ {
		if got := selectExecutionTarget(probs, legacy, false, false); got == 1 {
			t.Fatal("masked target selected")
		}
		if selectExecutionTarget(probs, greedy, false, true) != 2 {
			t.Fatal("greedy target did not choose largest probability")
		}
		selectExecutionAimMode(aim, legacy, false, false)
		if selectExecutionAimMode(aim, greedy, false, true) != 1 {
			t.Fatal("greedy aim mode did not choose largest probability")
		}
		for j := 0; j < 4; j++ {
			if legacy.NormFloat64() != greedy.NormFloat64() {
				t.Fatal("head choices shifted movement/aim random stream")
			}
		}
		for j := 0; j < 3; j++ {
			if legacy.Float64() != greedy.Float64() {
				t.Fatal("head choices shifted fire/pose/weapon random stream")
			}
		}
	}
	// Global deterministic execution draws no randomness; tied categories choose first.
	untouched := rand.New(rand.NewSource(123))
	deterministic := rand.New(rand.NewSource(123))
	if selectExecutionTarget([]float64{0, 0}, deterministic, true, false) != 0 ||
		selectExecutionAimMode([]float64{0, 0}, deterministic, true, false) != 0 ||
		untouched.Float64() != deterministic.Float64() {
		t.Fatal("deterministic tie or random draw contract changed")
	}
}
