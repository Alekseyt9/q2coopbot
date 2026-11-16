package learningenv

import (
	"encoding/json"
	"math"
	"strings"
	"testing"

	"q2coopbot/internal/policy"
	"q2coopbot/internal/quake"
)

func fragmentFixture() Fragment {
	id := policy.Identity{Map: "base1", Actor: 1, Life: 1, Connection: 2, Spawncount: 3, Frame: 1}
	next, logical := id, id
	next.Frame = 2
	logical.Frame = 120
	logical.Spawncount = 5
	return Fragment{Version: FragmentVersion, SourceDirectory: "source", BranchDirectory: "branch", ModelPath: "model", Split: "diagnostic", Family: "family", Model: "frozen", BaseSeed: 7, Sampling: policy.HJEPASamplingCheckpoint{Version: policy.HJEPASamplingBranchCheckpointVersion, Model: "frozen", Stream: 1, Source: logical}, NativeRoot: id,
		SourceSHA256: map[string]string{"trace": strings.Repeat("a", 64), "native": strings.Repeat("b", 64), "model": strings.Repeat("c", 64)},
		Rows:         []FragmentTransition{{ActorRaw: make([]float64, 8), Version: FragmentTransitionVersion, Index: 0, Identity: id, LogicalIdentity: logical, NextIdentity: next, Features: make([]float64, 871), NextFeatures: make([]float64, 871), NextHealth: 100, Action: policy.Action{Identity: id}, Sample: policy.Sample{Version: policy.HJEPAActorSampleVersion, SamplingSeed: 7}, Command: quake.UserCmd{Msec: 100}, Native: NativeStep{Spawncount: 3, Actor: 1, BeginFrame: 1, EndFrame: 2, Sequence: 1}, Truncated: true, Boundary: "horizon", BootstrapRequired: true}}}
}

func TestFragmentCutoffCannotBecomeZeroValueTerminal(t *testing.T) {
	f := fragmentFixture()
	if err := f.Validate(); err != nil {
		t.Fatal(err)
	}
	for _, mutate := range []func(*Fragment){
		func(f *Fragment) { f.Rows[0].BootstrapRequired = false },
		func(f *Fragment) {
			f.Rows[0].Terminal = true
			f.Rows[0].Truncated = false
			f.Rows[0].Boundary = "death"
			f.Rows[0].BootstrapRequired = false
			f.Rows[0].NextHealth = -1
		},
		func(f *Fragment) { f.Rows[0].LogicalIdentity.Frame++ },
		func(f *Fragment) { f.Rows[0].Native.EndFrame++ },
		func(f *Fragment) { f.Rows[0].Features[1] = math.NaN() },
		func(f *Fragment) { f.Split = "train" },
		func(f *Fragment) { f.OptimizerReady = true },
	} {
		data, _ := json.Marshal(f)
		var bad Fragment
		_ = json.Unmarshal(data, &bad)
		mutate(&bad)
		if bad.Validate() == nil {
			t.Fatal("invalid terminal/bootstrap/native binding accepted")
		}
	}
	f.Rows[0].Terminal = true
	f.Rows[0].Truncated = false
	f.Rows[0].BootstrapRequired = false
	f.Rows[0].Boundary = "death"
	f.Rows[0].NextHealth = -1
	f.Rows[0].NextFeatures = nil
	f.Rows[0].Events = []DamageEvent{{Map: "base1", Spawncount: 3, Frame: 2, Target: 1, HealthBefore: 100, HealthAfter: -1, Take: 101}}
	if err := f.Validate(); err != nil {
		t.Fatal("native death should close fragment", err)
	}
}
