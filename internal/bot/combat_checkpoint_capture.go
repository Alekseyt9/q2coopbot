package bot

import (
	"encoding/json"
	"fmt"
	"os"

	"q2coopbot/internal/harness/checkpoint"
	"q2coopbot/internal/policy"
)

func (c *Client) captureCombatCheckpointRequest(r checkpoint.CaptureRequest) error {
	if r.ID == c.checkpointCapturedID || c.latestFrame < r.Frame {
		return nil
	}
	s := c.planner.World.Snapshot
	capture := checkpoint.Capture{CaptureRequest: r, Participant: c.name, Self: s.Self, Health: s.Health, SelfEntity: c.decoder.PlayerNumber}
	if !c.testSynchronous || !c.testCombatBarrier || !c.testCombatGo || !c.framePaced || c.scenario != nil || c.session != nil ||
		c.latestFrame != r.Frame || s.Frame != r.Frame || c.lastMoveFrame != r.Frame-1 || s.Map != r.Map || c.spawncount != r.Generation || s.Health <= 0 {
		capture.Error = "combat checkpoint requires the exact living synchronous pre-decision boundary"
	} else {
		root := policy.Identity{Life: c.combatControl.life, Map: s.Map, Connection: c.connection, Spawncount: c.spawncount, Actor: c.decoder.PlayerNumber, Frame: s.Frame}
		state, err := c.combatControl.captureCombatCheckpoint(root, c.previous)
		if err != nil {
			capture.Error = err.Error()
		} else {
			capture.Combat, err = json.Marshal(state)
			if err != nil {
				return err
			}
		}
	}
	if err := checkpoint.WriteCapture(c.checkpointControl+".state.json", capture); err != nil {
		return err
	}
	c.checkpointCapturedID = r.ID
	if capture.Error != "" {
		return fmt.Errorf("%s", capture.Error)
	}
	c.checkpointHeld = &r
	return nil
}

func (c *Client) holdCombatCheckpoint() (bool, error) {
	r := c.checkpointHeld
	if r == nil {
		return false, nil
	}
	if c.latestFrame != r.Frame || c.lastMoveFrame != r.Frame-1 || c.spawncount != r.Generation || c.planner.World.Snapshot.Map != r.Map {
		return true, fmt.Errorf("combat checkpoint world/command advanced before release")
	}
	ready, err := checkpoint.SourceReleaseReady(c.checkpointControl+".released.json", *r)
	if os.IsNotExist(err) || checkpoint.ControlSharingError(err) {
		return true, nil
	}
	if err != nil {
		return true, err
	}
	if !ready {
		return true, nil
	}
	c.checkpointHeld = nil
	return false, nil
}
