package checkpoint

import (
	"encoding/json"
	"fmt"
	"path/filepath"
	"q2coopbot/internal/policy"
)

// RestoreReceipt is written before the client's first restored planner update.
type RestoreReceipt struct {
	Combat        *CombatRestoreReceipt `json:"combat,omitempty"`
	Version       int                   `json:"version"`
	SourceID      string                `json:"source_id"`
	Mode          string                `json:"mode"`
	Participant   string                `json:"participant"`
	SourceFrame   int                   `json:"source_frame"`
	Frame         int                   `json:"frame"`
	Generation    int                   `json:"generation"`
	SelfEntity    int                   `json:"self_entity"`
	Planner       json.RawMessage       `json:"planner,omitempty"`
	RunnerElapsed *int                  `json:"runner_elapsed,omitempty"`
}

type CombatRestoreReceipt struct {
	SamplingScope  string          `json:"sampling_scope,omitempty"`
	SamplingStream int64           `json:"sampling_stream,omitempty"`
	Version        string          `json:"version"`
	Root           policy.Identity `json:"decision_root"`
	Source         policy.Identity `json:"source_root"`
	Model          string          `json:"model"`
	PrefixFrame    int             `json:"prefix_frame"`
}

func verifyRestoreReceipts(dir string, manifest Manifest, b *Barrier) error {
	bindings, err := participantBindings(dir, manifest.Barrier)
	if err != nil {
		return err
	}
	if b.ID != manifest.Barrier.ID || b.Boundary != manifest.Barrier.Boundary || b.Map != manifest.Map || b.Generation == manifest.Barrier.Generation || len(b.Controls) != len(bindings) {
		return fmt.Errorf("restore barrier lineage/participant count mismatch")
	}
	seen := map[int]bool{}
	for _, control := range b.Controls {
		var r RestoreReceipt
		if err := readJSON(control+".restored.json", &r); err != nil {
			return fmt.Errorf("restore participant receipt unavailable: %w", err)
		}
		if r.Version != 1 || r.SourceID != b.ID || r.SourceFrame != manifest.Barrier.Frame || r.Frame != b.Frame || r.Generation != b.Generation || seen[r.SelfEntity] || bindings[r.SelfEntity] != r.Participant || (r.Mode != "resume" && r.Mode != "fresh" && r.Mode != "combat") {
			return fmt.Errorf("restore participant identity/anchor mismatch")
		}
		if b.Boundary == CombatPreDecisionBoundary {
			var source Capture
			if err := readJSON(filepath.Join(dir, "sidecar", r.Participant+".json"), &source); err != nil {
				return err
			}
			var state struct {
				Root     policy.Identity                `json:"decision_root"`
				Sampling policy.HJEPASamplingCheckpoint `json:"sampling"`
			}
			if json.Unmarshal(source.Combat, &state) != nil || r.Mode != "combat" || r.Combat == nil || len(r.Planner) > 0 || r.RunnerElapsed != nil ||
				r.Combat.SamplingScope != "" && r.Combat.SamplingScope != policy.HJEPASamplingFireOnlyScope && r.Combat.SamplingScope != policy.HJEPASamplingRootFireScope && r.Combat.SamplingScope != policy.HJEPASamplingRootJointScope ||
				r.Combat.Version != "hjepa_predecision_restore_v1" || r.Combat.Source != state.Root || r.Combat.Model != state.Sampling.Model ||
				r.Combat.SamplingStream < 0 || r.Combat.SamplingStream > 2147483647 || r.Combat.PrefixFrame != b.Frame-1 || r.Combat.Root.Map != b.Map || r.Combat.Root.Spawncount != b.Generation ||
				r.Combat.Root.Frame != b.Frame || r.Combat.Root.Actor != r.SelfEntity || r.Combat.Root.Life != state.Root.Life || r.Combat.Root.Connection < 1 {
				return fmt.Errorf("combat restore prefix/model/native anchor mismatch")
			}
			seen[r.SelfEntity] = true
			continue
		}
		if r.Mode == "combat" || r.Combat != nil {
			return fmt.Errorf("unexpected combat restore receipt")
		}
		var planner struct {
			Version int    `json:"version"`
			Map     string `json:"map"`
			Frame   int    `json:"captured_frame"`
		}
		if json.Unmarshal(r.Planner, &planner) != nil || planner.Version != 1 || planner.Map != b.Map || planner.Frame != b.Frame {
			return fmt.Errorf("restored planner anchor mismatch")
		}
		var source Capture
		if err := readJSON(filepath.Join(dir, "sidecar", r.Participant+".json"), &source); err != nil {
			return err
		}
		if r.Mode == "resume" && len(source.Runner) > 0 {
			var runner struct {
				Elapsed int `json:"elapsed_frames"`
			}
			if json.Unmarshal(source.Runner, &runner) != nil || r.RunnerElapsed == nil || *r.RunnerElapsed != runner.Elapsed {
				return fmt.Errorf("restored runner budget mismatch")
			}
		} else if r.RunnerElapsed != nil {
			return fmt.Errorf("unexpected restored runner")
		}
		seen[r.SelfEntity] = true
	}
	return nil
}
