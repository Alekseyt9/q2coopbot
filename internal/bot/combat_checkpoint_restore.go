package bot

import (
	"encoding/json"
	"fmt"

	"q2coopbot/internal/harness/checkpoint"
	"q2coopbot/internal/policy"
	"q2coopbot/internal/quake"
)

func (c *Client) restoreCombatCheckpointSnapshot(s quake.Snapshot) error {
	source := c.checkpointRestore
	if s.Frame < 1 {
		return nil
	}
	if source == nil || source.Boundary != checkpoint.CombatPreDecisionBoundary || len(source.Combat) == 0 || len(source.Planner) > 0 || len(source.Runner) > 0 ||
		!c.testSynchronous || !c.framePaced || !c.testCombatBarrier || c.scenario != nil || c.session != nil ||
		s.Frame != 1 || s.Map != source.Map || c.decoder.PlayerNumber != source.SelfEntity || c.spawncount == source.Generation || c.spawncount < 1 ||
		s.Health <= 0 || c.planner.observed {
		return fmt.Errorf("combat restore requires exact fresh living native load root")
	}
	var state combatCheckpoint
	if err := json.Unmarshal(source.Combat, &state); err != nil {
		return err
	}
	if state.Root.Frame != source.Frame || state.Root.Map != source.Map || state.Root.Spawncount != source.Generation || state.Root.Actor != source.SelfEntity {
		return fmt.Errorf("combat restore source boundary mismatch")
	}
	native := policy.Identity{Life: state.Root.Life, Map: s.Map, Frame: s.Frame, Connection: c.connection, Spawncount: c.spawncount, Actor: c.decoder.PlayerNumber}
	var fork []policy.HJEPASamplingCheckpoint
	if c.checkpointSamplingStream != nil {
		branch, err := state.Sampling.WithStream(*c.checkpointSamplingStream, c.checkpointSamplingScope)
		if err != nil {
			return err
		}
		fork = append(fork, branch)
	}
	previous, err := c.combatControl.restoreCombatCheckpoint(state, native, fork...)
	if err != nil {
		return err
	}
	// No teleport/give/test_combat_ready or fixture initialization after load.
	c.previous = previous
	c.testCombatGo, c.testTeleportSent, c.testPairReady, c.testHandGrenadeArmDone = true, true, true, true
	c.testCombatGoFrame, c.testTeleportSentFrame, c.testSetupHoldFrames = 1, -2, 0
	c.firstMoveFrame, c.lastMoveFrame = -1, 0
	r := checkpoint.CaptureRequest{Version: 1, Boundary: source.Boundary, ID: source.ID, Map: s.Map, Frame: 1, Generation: c.spawncount}
	c.checkpointHeld = &r
	receipt := checkpoint.RestoreReceipt{Version: 1, SourceID: source.ID, Mode: "combat", Participant: c.name, SourceFrame: source.Frame, Frame: 1, Generation: c.spawncount, SelfEntity: c.decoder.PlayerNumber,
		Combat: &checkpoint.CombatRestoreReceipt{Version: "hjepa_predecision_restore_v1", Root: native, Source: state.Root, Model: state.Sampling.Model, PrefixFrame: 0, SamplingStream: c.combatControl.provider.(*policy.HJEPA).SamplingStream(), SamplingScope: c.combatControl.provider.(*policy.HJEPA).SamplingScope()}}
	if err := checkpoint.WriteCapture(c.checkpointControl+".restored.json", receipt); err != nil {
		return err
	}
	c.checkpointRestored = true
	return nil
}
