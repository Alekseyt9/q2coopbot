package checkpoint

import (
	"encoding/json"
	"fmt"
	"path/filepath"
)

// RestoreReceipt is written before the client's first restored planner update.
type RestoreReceipt struct {
	Version       int             `json:"version"`
	SourceID      string          `json:"source_id"`
	Mode          string          `json:"mode"`
	Participant   string          `json:"participant"`
	SourceFrame   int             `json:"source_frame"`
	Frame         int             `json:"frame"`
	Generation    int             `json:"generation"`
	SelfEntity    int             `json:"self_entity"`
	Planner       json.RawMessage `json:"planner"`
	RunnerElapsed *int            `json:"runner_elapsed,omitempty"`
}

func verifyRestoreReceipts(dir string, manifest Manifest, b *Barrier) error {
	bindings, err := participantBindings(dir, manifest.Barrier)
	if err != nil {
		return err
	}
	if b.ID != manifest.Barrier.ID || b.Map != manifest.Map || b.Generation == manifest.Barrier.Generation || len(b.Controls) != len(bindings) {
		return fmt.Errorf("restore barrier lineage/participant count mismatch")
	}
	seen := map[int]bool{}
	for _, control := range b.Controls {
		var r RestoreReceipt
		if err := readJSON(control+".restored.json", &r); err != nil {
			return fmt.Errorf("restore participant receipt unavailable: %w", err)
		}
		if r.Version != 1 || r.SourceID != b.ID || r.SourceFrame != manifest.Barrier.Frame || r.Frame != b.Frame || r.Generation != b.Generation || seen[r.SelfEntity] || bindings[r.SelfEntity] != r.Participant || (r.Mode != "resume" && r.Mode != "fresh") {
			return fmt.Errorf("restore participant identity/anchor mismatch")
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
