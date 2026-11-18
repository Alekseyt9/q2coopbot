package main

import (
	"encoding/json"
	"fmt"
	"path/filepath"
	"strings"

	"q2coopbot/internal/harness/checkpoint"
	"q2coopbot/internal/policy"
)

// Select actual sampled coordinates for both root actions without changing the
// actor. These are stratified actions, not IID policy samples: fitting must use
// their frozen Bernoulli probability masses and a complete two-action pair.
func selectRootFireStreams(source string) (map[string]any, error) {
	var manifest checkpoint.Manifest
	manifestPath := filepath.Join(source, "combat-checkpoint", "manifest.json")
	if err := read(manifestPath, &manifest); err != nil {
		return nil, err
	}
	if manifest.Barrier == nil || manifest.Barrier.Boundary != checkpoint.CombatPreDecisionBoundary || len(manifest.Barrier.Participants) != 1 {
		return nil, fmt.Errorf("single combat checkpoint required")
	}
	name := strings.TrimSuffix(manifest.Barrier.Participants[0].Name, ".json")
	capture, err := checkpoint.ReadParticipant(filepath.Join(source, "combat-checkpoint"), name)
	if err != nil {
		return nil, err
	}
	var state struct {
		Root     policy.Identity                `json:"decision_root"`
		Sampling policy.HJEPASamplingCheckpoint `json:"sampling"`
	}
	if err := json.Unmarshal(capture.Combat, &state); err != nil {
		return nil, err
	}
	var config struct {
		Combat struct {
			Provider string `json:"provider_file"`
		} `json:"combat"`
	}
	configPath, tracePath := filepath.Join(source, "bot-config.json"), filepath.Join(source, "bot.jsonl")
	if err := read(configPath, &config); err != nil {
		return nil, err
	}
	p, err := policy.LoadHJEPA(config.Combat.Provider)
	if err != nil {
		return nil, err
	}
	trace, err := rows(tracePath)
	if err != nil {
		return nil, err
	}
	r, ok := trace[capture.Frame]
	if !ok || capture.Frame != state.Root.Frame || r.Combat == nil || r.Combat.Selection == nil || r.Combat.Selection.Owner != "provider" || r.Combat.Selection.Candidate == nil || r.Combat.Selection.ProviderVersion != p.Version() {
		return nil, fmt.Errorf("observed source actor root missing")
	}
	selected := map[bool]map[string]any{}
	for stream := int64(1); stream <= 4096; stream++ {
		p.ResetEpisode()
		fork, err := state.Sampling.WithStream(stream, policy.HJEPASamplingRootFireScope)
		if err != nil {
			return nil, err
		}
		if err := p.RestoreSamplingCheckpoint(fork, state.Root); err != nil {
			return nil, err
		}
		a, err := p.Decide(r.Combat.Observation)
		if err != nil {
			return nil, err
		}
		want := *r.Combat.Selection.Candidate
		want.Attack = a.Attack
		if a != want {
			return nil, fmt.Errorf("root stream altered non-fire source proposal")
		}
		if _, seen := selected[a.Attack]; !seen {
			selected[a.Attack] = map[string]any{"stream": stream, "attack": a.Attack, "joint_log_probability": p.LastSample().LogProbability}
		}
		if len(selected) == 2 {
			hashes := map[string]string{}
			for _, path := range []string{manifestPath, configPath, tracePath, config.Combat.Provider, filepath.Join(source, "combat-checkpoint", "sidecar", name+".json")} {
				h, err := digest(path)
				if err != nil {
					return nil, err
				}
				hashes[path] = h
			}
			return map[string]any{"accepted": true, "version": "hjepa_root_fire_stream_selection_v1", "sampling_scope": policy.HJEPASamplingRootFireScope, "selected": []map[string]any{selected[false], selected[true]}, "model": p.Version(), "source_checkpoint_id": capture.ID, "source_frame": capture.Frame, "candidates_examined": stream, "input_sha256": hashes, "scope": "Two action-stratified sampled root draws; common future policy RNG. Not IID on-policy samples or native continuation proof."}, nil
		}
	}
	return nil, fmt.Errorf("both root actions not sampled within4096declared coordinates")
}
