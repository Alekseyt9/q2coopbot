package bot

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"os"

	"q2coopbot/internal/harness/checkpoint"
	"q2coopbot/internal/policy"
)

func (c *Client) captureCheckpointRequest() error {
	if c.checkpointControl == "" {
		return nil
	}
	data, err := os.ReadFile(c.checkpointControl)
	if os.IsNotExist(err) {
		return nil
	}
	if err != nil {
		return err
	}
	if len(data) > 4096 {
		return fmt.Errorf("checkpoint request too large")
	}
	var request checkpoint.CaptureRequest
	d := json.NewDecoder(bytes.NewReader(data))
	d.DisallowUnknownFields()
	if err = d.Decode(&request); err != nil {
		return err
	}
	if d.Decode(new(any)) != io.EOF || request.Version != 1 || request.ID == "" || request.Frame < 1 {
		return fmt.Errorf("invalid checkpoint capture request")
	}
	if request.Phase != "" {
		validPhase := request.Phase == checkpoint.HandGrenadePrimedPhase && request.Weapon == "" ||
			(request.Phase == checkpoint.EquippedWeaponPhase || request.Phase == checkpoint.WeaponTriggerPhase) && request.Weapon != "" && policy.EquippedWeaponName(request.Weapon) == request.Weapon
		if request.Boundary != checkpoint.CombatPreDecisionBoundary || !validPhase {
			return fmt.Errorf("unsupported checkpoint capture phase")
		}
		if request.ID == c.checkpointCapturedID || c.latestFrame < request.Frame {
			return nil
		}
		s := c.planner.World.Snapshot
		if s.Map != request.Map || c.spawncount != request.Generation {
			return fmt.Errorf("checkpoint phase source identity changed")
		}
		if s.Health <= 0 || c.combatControl.life != 1 {
			return fmt.Errorf("checkpoint phase was not reached in the living first life")
		}
		if request.Phase == checkpoint.HandGrenadePrimedPhase && (s.GunFrame != 11 || policy.EquippedWeaponName(s.Weapon) != "Grenades") ||
			(request.Phase == checkpoint.EquippedWeaponPhase || request.Phase == checkpoint.WeaponTriggerPhase) && policy.EquippedWeaponName(s.Weapon) != request.Weapon {
			return nil
		}
		if request.Phase == checkpoint.WeaponTriggerPhase && !policy.ObservedWeaponTriggerPhase(s.Weapon, int(s.GunFrame), int(s.Ammo)) {
			return nil
		}
		// The native broker subsequently uses this resolved, already-held root.
		// No combat observation or action is consumed while selecting the event.
		request.Frame, request.Phase, request.Weapon = c.latestFrame, "", ""
	} else if request.Weapon != "" {
		return fmt.Errorf("checkpoint weapon predicate requires equipped weapon phase")
	}
	if request.Boundary == checkpoint.CombatPreDecisionBoundary {
		return c.captureCombatCheckpointRequest(request)
	}
	if request.Boundary != "" {
		return fmt.Errorf("unknown checkpoint request boundary")
	}
	if request.ID == c.checkpointCapturedID || c.latestFrame < request.Frame || c.lastMoveFrame < c.latestFrame {
		return nil
	}
	s := c.planner.World.Snapshot
	capture := checkpoint.Capture{CaptureRequest: request, Participant: c.name, Self: s.Self, Health: s.Health, SelfEntity: c.decoder.PlayerNumber}
	if s.Map != request.Map || c.spawncount != request.Generation || s.Frame != request.Frame {
		capture.Error = "client barrier anchor mismatch"
	} else {
		state, err := c.planner.CaptureCheckpoint()
		if err != nil {
			capture.Error = err.Error()
		} else {
			capture.Planner, err = json.Marshal(state)
			if err != nil {
				return err
			}
		}
		if capture.Error == "" && c.scenario != nil {
			state, err := c.scenario.CaptureCheckpoint()
			if err != nil {
				capture.Error = err.Error()
			} else {
				capture.Runner, err = json.Marshal(state)
				if err != nil {
					return err
				}
			}
		}
		if c.session != nil {
			capture.Error = "session phase checkpoint not yet supported"
		}
	}
	if err = checkpoint.WriteCapture(c.checkpointControl+".state.json", capture); err != nil {
		return err
	}
	c.checkpointCapturedID = request.ID
	return nil
}
