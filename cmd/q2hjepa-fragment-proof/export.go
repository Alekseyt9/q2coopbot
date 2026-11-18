package main

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"strings"

	"q2coopbot/internal/learningenv"
	"q2coopbot/internal/policy"
	"q2coopbot/internal/trainingepisodes"
)

func exportFragment(source, branch, out, planPath, repo string, n int) (map[string]any, error) {
	fork, err := replayFork(source, branch, n)
	if err != nil {
		return nil, err
	}
	var model policy.HJEPAFile
	if err := read(fork.modelPath, &model); err != nil {
		return nil, err
	}
	if model.PrimitiveSampling == nil {
		return nil, fmt.Errorf("sampled actor required")
	}
	f := learningenv.Fragment{Version: learningenv.FragmentVersion, Split: "diagnostic", Family: fork.sourceID, Model: fork.model, BaseSeed: model.PrimitiveSampling.Seed, Sampling: fork.sampling, NativeRoot: fork.nativeRoot, SourceSHA256: fork.hashes}
	if fork.sampling.Scope == policy.HJEPASamplingFireOnlyScope {
		f.Version = learningenv.FragmentFireVersion
	} else if fork.sampling.Scope == policy.HJEPASamplingRootFireScope {
		f.Version = learningenv.FragmentRootFireVersion
	} else if fork.sampling.Scope == policy.HJEPASamplingRootJointScope {
		f.Version = learningenv.FragmentRootJointVersion
	}
	f.SourceDirectory, f.BranchDirectory, f.ModelPath = source, branch, fork.modelPath
	if planPath != "" {
		if err := trainingepisodes.VerifyPlan(planPath, repo); err != nil {
			return nil, err
		}
		var plan trainingepisodes.Plan
		var fixture trainingepisodes.Instance
		if err := read(planPath, &plan); err != nil {
			return nil, err
		}
		if err := read(filepath.Join(source, "fixture.json"), &fixture); err != nil {
			return nil, err
		}
		matches := 0
		for _, task := range plan.Tasks {
			for _, instance := range task.Instances {
				if task.Split == "train" && reflect.DeepEqual(instance, fixture) {
					matches++
				}
			}
		}
		var receipt struct {
			Accepted bool                      `json:"accepted"`
			Fixture  trainingepisodes.Instance `json:"generated_fixture"`
			Start    struct {
				Confirmed bool `json:"confirmed"`
			} `json:"generated_start"`
		}
		if err := read(filepath.Join(source, "report.json"), &receipt); err != nil {
			return nil, err
		}
		if matches != 1 || fixture.Split != "train" || !receipt.Accepted || !receipt.Start.Confirmed || !reflect.DeepEqual(receipt.Fixture, fixture) || model.WeaponMode != "learned_masked_v1" || model.WeaponHead != policy.WeaponHeadVersion || model.Planner != "bc" || len(model.Motor) > 0 || model.WeaponOptionFrames != 0 || plan.DiagnosticNominalAim || plan.DiagnosticStoppedCrouch || plan.DiagnosticCheckedGroundRequest || plan.RecordCombatAdvice {
			return nil, fmt.Errorf("non-diagnostic full-actor TRAIN source not established")
		}
		f.Split, f.TrainingSourceVerified = "train", true
		f.PlanPath = planPath
		for _, p := range []string{planPath, filepath.Join(source, "fixture.json"), filepath.Join(source, "report.json"), plan.RegistryPath} {
			h, err := digest(p)
			if err != nil {
				return nil, err
			}
			f.SourceSHA256[p] = h
		}
	}
	log := strings.Join(fork.log, "\n")
	events, err := learningenv.ReadDamageEvents(strings.NewReader(log))
	if err != nil {
		return nil, err
	}
	native, err := learningenv.ReadNativeWindow(strings.NewReader(log), events)
	if err != nil {
		return nil, err
	}
	p, err := policy.LoadHJEPA(fork.modelPath)
	if err != nil {
		return nil, err
	}
	for i := 1; i <= fork.policyCount; i++ {
		r := fork.rows[i]
		next, ok := fork.rows[i+1]
		if !ok || next.Combat == nil {
			return nil, fmt.Errorf("fragment lacks next observed state at %d", i)
		}
		features, err := policy.HJEPAFeatures(r.Combat.Observation, policy.HJEPAContextFeatureVersion)
		if err != nil {
			return nil, err
		}
		var nextFeatures []float64
		if next.Combat.Observation.Health > 0 {
			nextFeatures, err = policy.HJEPAFeatures(next.Combat.Observation, policy.HJEPAContextFeatureVersion)
			if err != nil {
				return nil, err
			}
		}
		s := learningenv.Step{Observation: r.Combat.Observation, Next: &next.Combat.Observation, ClientSequence: uint32(r.Sequence)}
		step, err := native.Match(&s)
		if err != nil {
			return nil, err
		}
		logical := fork.sampling.Source
		logical.Frame += i - 1
		row := learningenv.FragmentTransition{Version: learningenv.FragmentTransitionVersion, Index: i - 1, Identity: r.Combat.Observation.Identity, LogicalIdentity: logical, NextIdentity: next.Combat.Observation.Identity, Features: features, NextFeatures: nextFeatures, NextHealth: next.Combat.Observation.Health, Action: *r.Combat.Selection.Candidate, Sample: *r.Combat.Selection.Sample, Command: r.Command, Interventions: r.Combat.Selection.Interventions, Native: *step, Events: []learningenv.DamageEvent{}}
		_, row.ActorRaw, _, err = p.Replay(features, []float64{0, 0, 0, 0, 0, 1, 0, 0})
		if err != nil {
			return nil, err
		}
		if model.WeaponSamplingSeed != nil {
			probs, err := p.WeaponDistribution(r.Combat.Observation)
			if err != nil {
				return nil, err
			}
			if row.Sample.Weapon < 0 || row.Sample.Weapon >= len(probs) || probs[row.Sample.Weapon] <= 0 {
				return nil, fmt.Errorf("sampled unavailable weapon")
			}
			row.WeaponLogProbability = math.Log(probs[row.Sample.Weapon])
		}
		row.PrimitiveLogProbability = row.Sample.LogProbability - row.WeaponLogProbability
		for _, index := range step.DamageIndexes {
			row.Events = append(row.Events, events[index])
		}
		if i == fork.policyCount {
			row.Boundary = fork.boundary
			row.Truncated = true
			row.BootstrapRequired = true
			if row.NextHealth <= 0 {
				row.Boundary = "death"
				row.Terminal = true
				row.Truncated = false
				row.BootstrapRequired = false
			}
		}
		f.Rows = append(f.Rows, row)
	}
	if err := f.Validate(); err != nil {
		return nil, err
	}
	// Re-read every bound input before publication. No partial file is a dataset.
	for path, want := range f.SourceSHA256 {
		got, err := digest(path)
		if err != nil || got != want {
			return nil, fmt.Errorf("fragment source changed: %s", path)
		}
	}
	data, err := json.Marshal(f)
	if err != nil {
		return nil, err
	}
	file, err := os.OpenFile(out, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
	if err != nil {
		return nil, err
	}
	_, err = file.Write(append(data, '\n'))
	closeErr := file.Close()
	if err != nil {
		return nil, err
	}
	if closeErr != nil {
		return nil, closeErr
	}
	h, err := digest(out)
	if err != nil {
		return nil, err
	}
	return map[string]any{"accepted": true, "version": f.Version, "rows": len(f.Rows), "split": f.Split, "training_source_verified": f.TrainingSourceVerified, "optimizer_ready": false, "boundary": f.Rows[len(f.Rows)-1].Boundary, "bootstrap_required": f.Rows[len(f.Rows)-1].BootstrapRequired, "data_sha256": h, "output": out}, nil
}
