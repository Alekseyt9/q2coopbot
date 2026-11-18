package main

import (
	"encoding/json"
	"fmt"
	"path/filepath"
	"reflect"
	"strings"

	"q2coopbot/internal/harness/checkpoint"
	"q2coopbot/internal/policy"
)

type forkData struct {
	stream          int64
	model, sourceID string
	policyCount     int
	validatedCount  int
	boundary        string
	sampling        policy.HJEPASamplingCheckpoint
	nativeRoot      policy.Identity
	modelPath       string
	rows            map[int]row
	log             []string
	hashes          map[string]string
}

func replayFork(source, branch string, n int) (forkData, error) {
	var result forkData
	var manifest checkpoint.Manifest
	if err := read(filepath.Join(source, "combat-checkpoint", "manifest.json"), &manifest); err != nil {
		return result, err
	}
	if manifest.Barrier == nil || len(manifest.Barrier.Participants) != 1 {
		return result, fmt.Errorf("single combat source required")
	}
	name := strings.TrimSuffix(manifest.Barrier.Participants[0].Name, ".json")
	capture, err := checkpoint.ReadParticipant(filepath.Join(source, "combat-checkpoint"), name)
	if err != nil {
		return result, err
	}
	var state struct {
		Root     policy.Identity                `json:"decision_root"`
		Sampling policy.HJEPASamplingCheckpoint `json:"sampling"`
	}
	if err := json.Unmarshal(capture.Combat, &state); err != nil {
		return result, err
	}
	var receipt checkpoint.RestoreReceipt
	if err := read(filepath.Join(branch, "control.json.restored.json"), &receipt); err != nil {
		return result, err
	}
	var load, release checkpoint.Result
	if err := read(filepath.Join(branch, "load-result.json"), &load); err != nil {
		return result, err
	}
	if err := read(filepath.Join(branch, "release-result.json"), &release); err != nil {
		return result, err
	}
	if capture.Boundary != checkpoint.CombatPreDecisionBoundary || receipt.Combat == nil || receipt.Mode != "combat" || receipt.SourceID != capture.ID || receipt.SourceFrame != capture.Frame || receipt.Participant != capture.Participant ||
		receipt.Combat.Source != state.Root || receipt.Combat.Model != state.Sampling.Model || receipt.Combat.Root.Frame != 1 || receipt.Combat.PrefixFrame != 0 ||
		receipt.Combat.Root.Map != capture.Map || receipt.Combat.Root.Actor != capture.SelfEntity || receipt.Combat.Root.Spawncount != receipt.Generation || receipt.Combat.Root.Life != state.Root.Life ||
		!load.WorldUnadvanced || !load.CombatRestored || !load.RNGRestored || load.State != "load_held" || load.LoadAnchor == nil || load.LoadAnchor.ID != capture.ID || load.LoadAnchor.Generation != receipt.Generation || load.LoadAnchor.Frame != 1 ||
		release.State != "release_verified" || !release.BarrierVerified {
		return result, fmt.Errorf("fork native/controller/source lineage missing")
	}
	var config struct {
		Combat struct {
			Provider string `json:"provider_file"`
		} `json:"combat"`
		Test struct {
			Stream *int64 `json:"checkpoint_sampling_stream"`
			Scope  string `json:"checkpoint_sampling_scope"`
		} `json:"test"`
	}
	if err := read(filepath.Join(branch, "bot-config.json"), &config); err != nil {
		return result, err
	}
	if config.Test.Stream == nil || *config.Test.Stream != receipt.Combat.SamplingStream || config.Test.Scope != receipt.Combat.SamplingScope {
		return result, fmt.Errorf("fork stream not explicitly bound by config")
	}
	p, err := policy.LoadHJEPA(config.Combat.Provider)
	if err != nil {
		return result, err
	}
	state.Sampling, err = state.Sampling.WithStream(*config.Test.Stream, config.Test.Scope)
	if err != nil {
		return result, err
	}
	if err := p.RestoreSamplingCheckpoint(state.Sampling, receipt.Combat.Root); err != nil {
		return result, err
	}
	result.sampling, result.nativeRoot, result.modelPath = state.Sampling, receipt.Combat.Root, config.Combat.Provider
	result.rows, err = rows(filepath.Join(branch, "bot.jsonl"))
	if err != nil {
		return result, err
	}
	result.log, err = lines(filepath.Join(branch, "server.log"))
	if err != nil {
		return result, err
	}
	result.policyCount = n
	result.validatedCount = n
	result.boundary = "horizon"
	for i := 0; i < n; i++ {
		r, ok := result.rows[i+1]
		if !ok || r.Combat == nil || r.Combat.Selection == nil {
			return result, fmt.Errorf("fork observation/selection missing at %d", i)
		}
		id := receipt.Combat.Root
		id.Frame += i
		if r.Generation != receipt.Generation || r.Actor != capture.SelfEntity || r.Combat.Observation.Identity != id {
			return result, fmt.Errorf("fork observed identity/model mismatch at %d", i)
		}
		if r.Combat.Observation.Health <= 0 {
			result.validatedCount = i
			if i < result.policyCount {
				result.policyCount = i
				result.boundary = "death"
			}
			break
		}
		if err := applied(result.log, r); err != nil {
			return result, err
		}
		if i < result.policyCount && r.Combat.Selection.Owner != "provider" {
			result.policyCount = i
			result.boundary = "control_transfer:" + r.Combat.Selection.Fallback
		}
		// Retain and verify the native tail, but never admit rules actions as
		// sampled-policy decisions or continue a policy prefix across transfer.
		if i >= result.policyCount {
			continue
		}
		if r.Combat.Selection.Candidate == nil || r.Combat.Selection.Sample == nil || r.Combat.Selection.ProviderVersion != state.Sampling.Model {
			return result, fmt.Errorf("fork sampled policy receipt missing at %d", i)
		}
		a, err := p.Decide(r.Combat.Observation)
		if err != nil {
			return result, err
		}
		if a != *r.Combat.Selection.Candidate || !reflect.DeepEqual(p.LastSample(), r.Combat.Selection.Sample) {
			return result, fmt.Errorf("fork actor/density/draw replay mismatch at %d", i)
		}
	}
	result.stream, result.model, result.sourceID = *config.Test.Stream, state.Sampling.Model, capture.ID
	result.hashes = map[string]string{}
	for _, path := range []string{filepath.Join(branch, "bot.jsonl"), filepath.Join(branch, "server.log"), filepath.Join(branch, "bot-config.json"), filepath.Join(branch, "control.json.restored.json"), filepath.Join(source, "combat-checkpoint", "manifest.json"), config.Combat.Provider} {
		h, err := digest(path)
		if err != nil {
			return result, err
		}
		result.hashes[path] = h
	}
	return result, nil
}

func verifyForkPair(source, branch, other string, n int, same bool) (map[string]any, error) {
	a, err := replayFork(source, branch, n)
	if err != nil {
		return nil, err
	}
	b, err := replayFork(source, other, n)
	if err != nil {
		return nil, err
	}
	if a.sourceID != b.sourceID || a.model != b.model || a.sampling.Scope != b.sampling.Scope || (a.stream == b.stream) != same {
		return nil, fmt.Errorf("fork source/model/stream comparison mismatch")
	}
	if same && (a.policyCount != b.policyCount || a.boundary != b.boundary || a.validatedCount != b.validatedCount) {
		return nil, fmt.Errorf("repeated fork policy boundary changed")
	}
	count := min(a.policyCount, b.policyCount)
	if count < 2 {
		return nil, fmt.Errorf("insufficient sampled-policy prefix")
	}
	differences := 0
	for i := 1; i <= min(a.validatedCount, b.validatedCount); i++ {
		x, y := a.rows[i], b.rows[i]
		fx, err := policy.HJEPAFeatures(x.Combat.Observation, policy.HJEPAContextFeatureVersion)
		if err != nil {
			return nil, err
		}
		fy, err := policy.HJEPAFeatures(y.Combat.Observation, policy.HJEPAContextFeatureVersion)
		if err != nil {
			return nil, err
		}
		if (i == 1 || same) && !reflect.DeepEqual(fx, fy) {
			return nil, fmt.Errorf("fork common world/root features differ at %d", i)
		}
		if i <= count {
			px, py := *x.Combat.Selection.Candidate, *y.Combat.Selection.Candidate
			px.Identity = py.Identity
			if i == 1 && (a.sampling.Scope == policy.HJEPASamplingFireOnlyScope || a.sampling.Scope == policy.HJEPASamplingRootFireScope) {
				px.Attack = py.Attack
				if px != py || x.Combat.Selection.Sample.Latent != y.Combat.Selection.Sample.Latent || x.Combat.Selection.Sample.Vertical != y.Combat.Selection.Sample.Vertical || x.Combat.Selection.Sample.Weapon != y.Combat.Selection.Sample.Weapon {
					return nil, fmt.Errorf("fire-only fork changed non-fire root proposal/draws")
				}
				px.Attack = x.Combat.Selection.Candidate.Attack
			}
			if px != py {
				differences++
			}
			if same && (px != py || !reflect.DeepEqual(x.Combat.Selection.Sample, y.Combat.Selection.Sample)) {
				return nil, fmt.Errorf("repeated fork policy proposal changed at %d", i)
			}
		}
		if same && x.Command != y.Command {
			return nil, fmt.Errorf("repeated fork native tail changed at %d", i)
		}
	}
	if same && !reflect.DeepEqual(effects(a.log, a.rows[1].Generation, 1, a.validatedCount), effects(b.log, b.rows[1].Generation, 1, b.validatedCount)) {
		return nil, fmt.Errorf("repeated fork native effects changed")
	}
	if !same && differences == 0 {
		return nil, fmt.Errorf("different streams produced no proposal differences")
	}
	for p, h := range b.hashes {
		a.hashes[p] = h
	}
	return map[string]any{"accepted": true, "proof": "native_hjepa_fragment_fork_v1", "sampling_scope": a.sampling.Scope, "root_non_fire_draws_equal": a.sampling.Scope == policy.HJEPASamplingFireOnlyScope || a.sampling.Scope == policy.HJEPASamplingRootFireScope, "future_policy_rng_common": a.sampling.Scope == policy.HJEPASamplingRootFireScope || a.sampling.Scope == policy.HJEPASamplingRootJointScope, "same_stream_repeat": same, "streams": []int64{a.stream, b.stream}, "requested_native_frames": n, "validated_native_frames": []int{a.validatedCount, b.validatedCount}, "sampled_policy_decisions": []int{a.policyCount, b.policyCount}, "policy_boundaries": []string{a.boundary, b.boundary}, "proposal_differences": differences, "root_features_equal": 871, "exact_actor_and_density_replay": true, "native_single_steps_verified": a.validatedCount + b.validatedCount, "model": a.model, "source_checkpoint_id": a.sourceID, "input_sha256": a.hashes, "fragment_training": false, "promotion": false}, nil
}
