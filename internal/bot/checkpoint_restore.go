package bot

import (
	"encoding/json"
	"fmt"

	"q2coopbot/internal/harness"
	"q2coopbot/internal/harness/checkpoint"
	"q2coopbot/internal/quake"
)

type checkpointRestoreReceipt struct {
	Mode          string            `json:"mode"`
	Participant   string            `json:"participant"`
	SourceFrame   int               `json:"source_frame"`
	Frame         int               `json:"frame"`
	Generation    int               `json:"generation"`
	SelfEntity    int               `json:"self_entity"`
	Planner       PlannerCheckpoint `json:"planner"`
	RunnerElapsed *int              `json:"runner_elapsed,omitempty"`
}

// Called before the first planner update or scripted command after signon.
func (c *Client) restoreCheckpointSnapshot(s quake.Snapshot) error {
	source := c.checkpointRestore
	if source == nil || c.checkpointRestored {
		return nil
	}
	if s.Map != source.Map || c.decoder.PlayerNumber != source.SelfEntity || c.spawncount == source.Generation {
		return fmt.Errorf("checkpoint restore native identity/generation mismatch")
	}
	if s.Frame < 1 || !s.OnGround {
		return nil
	}
	if s.Health <= 0 {
		return fmt.Errorf("checkpoint restore requires a living actor")
	}
	if c.checkpointMode != "resume" && c.checkpointMode != "fresh" {
		return fmt.Errorf("unknown checkpoint mode")
	}
	if c.checkpointMode == "fresh" && c.scenario != nil {
		return fmt.Errorf("fresh scenario would replay placement; use resume or omit scenario")
	}
	next := *c.planner
	var runner *harness.Runner
	receipt := checkpointRestoreReceipt{Mode: c.checkpointMode, Participant: c.name, SourceFrame: source.Frame, Frame: s.Frame, Generation: c.spawncount, SelfEntity: c.decoder.PlayerNumber}
	if c.checkpointMode == "resume" {
		if (len(source.Runner) > 0) != (c.scenario != nil) {
			return fmt.Errorf("runner checkpoint/config mismatch")
		}
		if c.scenario != nil {
			var state harness.RunnerCheckpoint
			if err := json.Unmarshal(source.Runner, &state); err != nil {
				return err
			}
			var err error
			runner, err = harness.RestoreRunnerCheckpoint(c.scenario.Scenario, state, harness.Input{Frame: s.Frame, Generation: c.spawncount, Map: s.Map, Self: s.Self, OnGround: s.OnGround, Health: s.Health})
			if err != nil {
				return err
			}
			receipt.RunnerElapsed = &state.Elapsed
		}
		var state PlannerCheckpoint
		if err := json.Unmarshal(source.Planner, &state); err != nil {
			return err
		}
		if err := next.RestoreCheckpoint(state, s, c.root); err != nil {
			return err
		}
	} else {
		next.setMap(s.Map, c.root)
		next.World.Snapshot = s
	}
	var err error
	receipt.Planner, err = next.CaptureCheckpoint()
	if err != nil {
		return err
	}
	if err = checkpoint.WriteCapture(c.checkpointControl+".restored.json", receipt); err != nil {
		return err
	}
	c.planner = &next
	if runner != nil {
		c.scenario = runner
	}
	c.checkpointRestored = true
	return nil
}
