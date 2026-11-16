package learningenv

import (
	"fmt"
	"math"
	"strings"

	"q2coopbot/internal/policy"
	"q2coopbot/internal/quake"
)

const FragmentVersion = "hjepa_native_fragment_v2"
const FragmentTransitionVersion = "hjepa_native_fragment_transition_v2"

// Fragment is a conditional rollout from one exact native checkpoint. It is
// deliberately distinct from a reset episode and carries no invented tail value.
type Fragment struct {
	SourceDirectory        string                         `json:"source_directory"`
	BranchDirectory        string                         `json:"branch_directory"`
	ModelPath              string                         `json:"behavior_model"`
	PlanPath               string                         `json:"training_plan,omitempty"`
	Version                string                         `json:"version"`
	Split                  string                         `json:"split"`
	TrainingSourceVerified bool                           `json:"training_source_verified"`
	OptimizerReady         bool                           `json:"optimizer_ready"`
	Family                 string                         `json:"checkpoint_family"`
	Model                  string                         `json:"model_version"`
	BaseSeed               int64                          `json:"base_sampling_seed"`
	Sampling               policy.HJEPASamplingCheckpoint `json:"sampling_checkpoint"`
	NativeRoot             policy.Identity                `json:"native_root"`
	SourceSHA256           map[string]string              `json:"sources_sha256"`
	Rows                   []FragmentTransition           `json:"transitions"`
}

type FragmentTransition struct {
	ActorRaw                []float64             `json:"actor_raw"`
	PrimitiveLogProbability float64               `json:"primitive_log_probability"`
	WeaponLogProbability    float64               `json:"weapon_log_probability"`
	Version                 string                `json:"version"`
	Index                   int                   `json:"index"`
	Identity                policy.Identity       `json:"identity"`
	LogicalIdentity         policy.Identity       `json:"logical_sampling_identity"`
	NextIdentity            policy.Identity       `json:"next_identity"`
	Features                []float64             `json:"features"`
	NextFeatures            []float64             `json:"next_features,omitempty"`
	NextHealth              int16                 `json:"next_health"`
	Action                  policy.Action         `json:"proposal"`
	Sample                  policy.Sample         `json:"sample"`
	Command                 quake.UserCmd         `json:"sent_command"`
	Interventions           []policy.Intervention `json:"interventions,omitempty"`
	Native                  NativeStep            `json:"native_step"`
	Events                  []DamageEvent         `json:"native_effects"`
	Terminal                bool                  `json:"terminal"`
	Truncated               bool                  `json:"truncated"`
	Boundary                string                `json:"boundary,omitempty"`
	BootstrapRequired       bool                  `json:"bootstrap_required"`
}

func (f Fragment) Validate() error {
	if f.Version != FragmentVersion || f.SourceDirectory == "" || f.BranchDirectory == "" || len(f.SourceSHA256[f.ModelPath]) != 64 || f.Family == "" || f.Model == "" || f.Sampling.Model != f.Model ||
		f.Sampling.Stream < 0 || f.Sampling.Stream > 2147483647 || f.NativeRoot.Frame != 1 || f.NativeRoot.Life < 1 || f.BaseSeed < 0 ||
		len(f.Rows) < 1 || len(f.Rows) > 100 || len(f.SourceSHA256) < 3 || f.Split != "train" && f.Split != "diagnostic" ||
		f.TrainingSourceVerified != (f.Split == "train") || f.OptimizerReady {
		return fmt.Errorf("fragment header/admission rejected")
	}
	if f.TrainingSourceVerified && (f.PlanPath == "" || len(f.SourceSHA256[f.PlanPath]) != 64) || !f.TrainingSourceVerified && f.PlanPath != "" {
		return fmt.Errorf("fragment TRAIN plan binding missing")
	}
	if f.Sampling.Version != policy.HJEPASamplingBranchCheckpointVersion && (f.Sampling.Version != policy.HJEPASamplingCheckpointVersion || f.Sampling.Stream != 0) {
		return fmt.Errorf("fragment sampling version rejected")
	}
	for _, hash := range f.SourceSHA256 {
		if len(hash) != 64 {
			return fmt.Errorf("fragment source hash missing")
		}
	}
	for i, r := range f.Rows {
		native, logical := f.NativeRoot, f.Sampling.Source
		native.Frame += i
		logical.Frame += i
		if r.Version != FragmentTransitionVersion || r.Index != i || r.Identity != native || r.LogicalIdentity != logical || r.Action.Identity != native ||
			!policy.SameLife(r.NextIdentity, native) || r.NextIdentity.Frame != native.Frame+1 ||
			r.Native.Spawncount != native.Spawncount || r.Native.Actor != native.Actor || r.Native.BeginFrame != native.Frame || r.Native.EndFrame != r.NextIdentity.Frame || r.Native.Sequence == 0 ||
			r.Sample.Version != policy.HJEPAActorSampleVersion || r.Sample.SamplingSeed != f.BaseSeed || r.Command.Msec != 100 {
			return fmt.Errorf("fragment transition/native/draw mismatch at %d", i)
		}
		finite := func(x []float64) bool {
			if len(x) != 871 {
				return false
			}
			for _, v := range x {
				if math.IsNaN(v) || math.IsInf(v, 0) {
					return false
				}
			}
			return true
		}
		if !finite(r.Features) || r.NextHealth > 0 && !finite(r.NextFeatures) || math.IsNaN(r.Sample.LogProbability) || math.IsInf(r.Sample.LogProbability, 0) {
			return fmt.Errorf("fragment nonfinite/missing feature/density")
		}
		if len(r.ActorRaw) != 8 || math.IsNaN(r.PrimitiveLogProbability+r.WeaponLogProbability) || math.Abs(r.PrimitiveLogProbability+r.WeaponLogProbability-r.Sample.LogProbability) > 1e-9 {
			return fmt.Errorf("fragment joint density factorization missing")
		}
		for _, v := range r.ActorRaw {
			if math.IsNaN(v) || math.IsInf(v, 0) {
				return fmt.Errorf("fragment actor logits nonfinite")
			}
		}
		death := false
		for _, e := range r.Events {
			if e.Map != native.Map || e.Spawncount != native.Spawncount || e.Frame < r.Native.BeginFrame || e.Frame > r.Native.EndFrame || e.HealthBefore-e.Take != e.HealthAfter {
				return fmt.Errorf("fragment effect outside causal native step")
			}
			if e.Target == native.Actor && e.HealthBefore > 0 && e.HealthAfter <= 0 {
				death = true
			}
		}
		last := i == len(f.Rows)-1
		if !last && (r.Terminal || r.Truncated || r.Boundary != "" || r.BootstrapRequired || r.NextHealth <= 0) {
			return fmt.Errorf("fragment crosses boundary")
		}
		if last {
			if r.Terminal {
				if r.Truncated || r.Boundary != "death" || r.BootstrapRequired || r.NextHealth > 0 || !death {
					return fmt.Errorf("fragment terminal death unproved")
				}
			} else if !r.Truncated || !r.BootstrapRequired || r.NextHealth <= 0 || r.Boundary != "horizon" && !strings.HasPrefix(r.Boundary, "control_transfer:") {
				return fmt.Errorf("fragment cutoff must retain bootstrap requirement")
			}
		}
	}
	return nil
}
