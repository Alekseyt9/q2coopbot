package bot

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"

	"q2coopbot/internal/harness/checkpoint"
)

func TestObservedPhaseCaptureResolvesAndHoldsExactLaterRoot(t *testing.T) {
	for _, test := range []struct{ phase, wanted, waiting, matched string }{
		{checkpoint.HandGrenadePrimedPhase, "", "Grenades", "Grenades"},
		{checkpoint.EquippedWeaponPhase, "Blaster", "Grenades", "models/weapons/v_blast/tris.md2"},
		{checkpoint.EquippedWeaponPhase, "Railgun", "Blaster", "models/weapons/v_rail/tris.md2"},
		{checkpoint.EquippedWeaponPhase, "Machinegun", "Grenades", "Machinegun"},
		{checkpoint.WeaponTriggerPhase, "Railgun", "Railgun", "models/weapons/v_rail/tris.md2"},
	} {
		t.Run(test.phase+test.wanted, func(t *testing.T) {
			c := checkpointTestClient(t, combatCheckpointModel(t), 2, 3)
			c.framePaced, c.testCombatBarrier = true, true
			c.name = "PhaseFragmentBot"
			c.checkpointControl = filepath.Join(t.TempDir(), "phase-control")
			for step := 0; step < 5; step++ {
				c.planner.World.Snapshot = checkpointTestSnapshot(170+step, step)
				c.planner.World.Updated = time.Now()
				o := c.combatObservation(time.Now())
				c.combatControl.engagement.observe(o, nil)
			}
			r := checkpoint.CaptureRequest{Version: 1, Boundary: checkpoint.CombatPreDecisionBoundary, Phase: test.phase, Weapon: test.wanted, ID: "phase", Map: "base1", Frame: 170, Generation: 3}
			c.latestFrame, c.lastMoveFrame = 175, 174
			c.planner.World.Snapshot = checkpointTestSnapshot(175, 5)
			c.planner.World.Snapshot.GunFrame = 10
			c.planner.World.Snapshot.Weapon = test.waiting
			if test.phase == checkpoint.WeaponTriggerPhase {
				c.planner.World.Snapshot.GunFrame = 19
				c.planner.World.Snapshot.Ammo = 20
			}
			if err := checkpoint.WriteCapture(c.checkpointControl, r); err != nil {
				t.Fatal(err)
			}
			if err := c.captureCheckpointRequest(); err != nil {
				t.Fatal(err)
			}
			if _, err := os.Stat(c.checkpointControl + ".state.json"); !os.IsNotExist(err) {
				t.Fatal("phase captured too early", err)
			}
			if c.combatControl.last.Frame != 174 {
				t.Fatal("phase selection consumed an observation")
			}
			o := c.combatObservation(time.Now())
			c.combatControl.engagement.observe(o, nil)
			c.latestFrame, c.lastMoveFrame = 176, 175
			c.planner.World.Snapshot = checkpointTestSnapshot(176, 6)
			c.planner.World.Snapshot.Weapon = test.matched
			if test.phase == checkpoint.WeaponTriggerPhase {
				c.planner.World.Snapshot.GunFrame = 20
				c.planner.World.Snapshot.Ammo = 20
			}
			if err := c.captureCheckpointRequest(); err != nil {
				t.Fatal(err)
			}
			var capture checkpoint.Capture
			b, err := os.ReadFile(c.checkpointControl + ".state.json")
			if err != nil {
				t.Fatal(err)
			}
			if err := json.Unmarshal(b, &capture); err != nil {
				t.Fatal(err)
			}
			if capture.Error != "" || capture.Frame != 176 || capture.Phase != "" || capture.Weapon != "" || c.combatControl.last.Frame != 175 || c.lastMoveFrame != 175 {
				t.Fatal("phase did not resolve before the next observation/action", capture.Error)
			}
			if held, err := c.holdCombatCheckpoint(); err != nil || !held {
				t.Fatal("resolved root not held", err)
			}
			if err := checkpoint.WriteCapture(c.checkpointControl+".released.json", checkpoint.SourceRelease{CaptureRequest: capture.CaptureRequest, State: "native_released"}); err != nil {
				t.Fatal(err)
			}
			if held, err := c.holdCombatCheckpoint(); err != nil || held {
				t.Fatal("resolved native release not accepted", err)
			}
		})
	}
}

func TestCombatCaptureHoldsBeforeObservationUntilExactRelease(t *testing.T) {
	path := combatCheckpointModel(t)
	c := checkpointTestClient(t, path, 2, 3)
	c.framePaced, c.testCombatBarrier = true, true
	c.name = "FragmentBot"
	c.checkpointControl = filepath.Join(t.TempDir(), "control")
	for step := 0; step < 5; step++ {
		c.planner.World.Snapshot = checkpointTestSnapshot(170+step, step)
		c.planner.World.Updated = time.Now()
		o := c.combatObservation(time.Now())
		c.combatControl.engagement.observe(o, nil)
	}
	r := checkpoint.CaptureRequest{Version: 1, Boundary: checkpoint.CombatPreDecisionBoundary, ID: "combat_a", Map: "base1", Frame: 175, Generation: 3}
	c.latestFrame, c.lastMoveFrame = 175, 174
	c.planner.World.Snapshot = checkpointTestSnapshot(175, 5)
	if err := checkpoint.WriteCapture(c.checkpointControl, r); err != nil {
		t.Fatal(err)
	}
	if err := c.captureCheckpointRequest(); err != nil {
		t.Fatal(err)
	}
	if c.combatControl.last.Frame != 174 || c.lastMoveFrame != 174 {
		t.Fatal("root observed or commanded during capture")
	}
	data, err := os.ReadFile(c.checkpointControl + ".state.json")
	if err != nil {
		t.Fatal(err)
	}
	var captured checkpoint.Capture
	if err = json.Unmarshal(data, &captured); err != nil {
		t.Fatal(err)
	}
	if captured.Error != "" || len(captured.Combat) == 0 || len(captured.Planner) != 0 {
		t.Fatal("combat payload missing or confused with idle planner")
	}
	for i := 0; i < 3; i++ {
		if held, err := c.holdCombatCheckpoint(); err != nil || !held {
			t.Fatal("did not hold before release", held, err)
		}
		if err := c.captureCheckpointRequest(); err != nil {
			t.Fatal(err)
		}
	}
	release := checkpoint.SourceRelease{CaptureRequest: r, State: "native_released"}
	release.ID = "old_capture"
	if err := checkpoint.WriteCapture(c.checkpointControl+".released.json", release); err != nil {
		t.Fatal(err)
	}
	if held, err := c.holdCombatCheckpoint(); err != nil || !held {
		t.Fatal("stale receipt released root")
	}
	release.ID = r.ID
	release.Frame++
	if err := checkpoint.WriteCapture(c.checkpointControl+".released.json", release); err != nil {
		t.Fatal(err)
	}
	if _, err := c.holdCombatCheckpoint(); err == nil {
		t.Fatal("wrong anchor accepted")
	}
	release.Frame = r.Frame
	if err := checkpoint.WriteCapture(c.checkpointControl+".released.json", release); err != nil {
		t.Fatal(err)
	}
	if held, err := c.holdCombatCheckpoint(); err != nil || held {
		t.Fatal("exact release not accepted", held, err)
	}
	if err := c.captureCheckpointRequest(); err != nil {
		t.Fatal(err)
	}
	if c.checkpointHeld != nil || c.combatControl.last.Frame != 174 {
		t.Fatal("root recaptured or consumed")
	}
	// Generic resume must not silently discard the combat payload.
	c.checkpointRestore = &captured
	if err := c.restoreCheckpointSnapshot(c.planner.World.Snapshot); err == nil {
		t.Fatal("combat payload admitted by legacy resume")
	}
}

func TestCombatCaptureRejectsRootCommandAlreadySent(t *testing.T) {
	c := checkpointTestClient(t, combatCheckpointModel(t), 2, 3)
	c.framePaced, c.testCombatBarrier = true, true
	c.checkpointControl = filepath.Join(t.TempDir(), "control")
	c.latestFrame, c.lastMoveFrame = 175, 175
	c.planner.World.Snapshot = checkpointTestSnapshot(175, 5)
	r := checkpoint.CaptureRequest{Version: 1, Boundary: checkpoint.CombatPreDecisionBoundary, ID: "late", Map: "base1", Frame: 175, Generation: 3}
	if err := c.captureCombatCheckpointRequest(r); err == nil {
		t.Fatal("already sent root admitted")
	}
	if c.checkpointHeld != nil {
		t.Fatal("failed capture became resumable")
	}
}
