package main

import (
	"fmt"
	"path/filepath"
	"reflect"

	"q2coopbot/internal/harness/checkpoint"
	"q2coopbot/internal/policy"
)

// Counterfactual streams may change the first action, but never its inputs.
func verifySourceRoot(source, branch string) (map[string]any, error) {
	fork, err := replayFork(source, branch, 1)
	if err != nil {
		return nil, err
	}
	if fork.policyCount != 1 {
		return nil, fmt.Errorf("loaded root not controlled by sampled actor")
	}
	var manifest checkpoint.Manifest
	if err := read(filepath.Join(source, "combat-checkpoint", "manifest.json"), &manifest); err != nil {
		return nil, err
	}
	if manifest.Barrier == nil {
		return nil, fmt.Errorf("source root missing")
	}
	path := filepath.Join(source, "bot.jsonl")
	actual, err := rows(path)
	if err != nil {
		return nil, err
	}
	r, ok := actual[manifest.Barrier.Frame]
	if !ok || r.Combat == nil || r.Combat.Selection == nil || r.Combat.Selection.Owner != "provider" || r.Combat.Selection.ProviderVersion != fork.model {
		return nil, fmt.Errorf("original source actor observation missing")
	}
	var model policy.HJEPAFile
	if err := read(fork.modelPath, &model); err != nil {
		return nil, err
	}
	a, err := policy.HJEPAFeatures(r.Combat.Observation, model.Features)
	if err != nil {
		return nil, err
	}
	b, err := policy.HJEPAFeatures(fork.rows[1].Combat.Observation, model.Features)
	if err != nil {
		return nil, err
	}
	if !reflect.DeepEqual(a, b) {
		return nil, fmt.Errorf("loaded root features differ from original observed source")
	}
	h, err := digest(path)
	if err != nil {
		return nil, err
	}
	fork.hashes[path] = h
	return map[string]any{"accepted": true, "proof": "native_original_source_root_features_v1", "source_frame": manifest.Barrier.Frame, "loaded_frame": 1, "features_equal": len(a), "sampling_stream": fork.stream, "source_checkpoint_id": fork.sourceID, "model": fork.model, "input_sha256": fork.hashes, "scope": "Original source versus loaded counterfactual root inputs; actual first sampled action/density and native command replay verified, no later trajectory equality claim"}, nil
}
