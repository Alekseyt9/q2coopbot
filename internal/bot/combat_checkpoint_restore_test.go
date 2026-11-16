package bot

import (
	"encoding/json"
	"path/filepath"
	"testing"
	"time"

	"q2coopbot/internal/harness/checkpoint"
	"q2coopbot/internal/policy"
)

func TestCombatRestoreWaitsForReleaseAndSkipsFixtureSetup(t *testing.T) {
	path := combatCheckpointModel(t)
	source := checkpointTestClient(t, path, 2, 3)
	for step := 0; step < 5; step++ {
		source.planner.World.Snapshot = checkpointTestSnapshot(170+step, step)
		source.planner.World.Updated = time.Now()
		o := source.combatObservation(time.Now())
		source.combatControl.engagement.observe(o, nil)
	}
	root := source.combatControl.last
	root.Frame++
	state, err := source.combatControl.captureCombatCheckpoint(root, source.previous)
	if err != nil {
		t.Fatal(err)
	}
	raw, _ := json.Marshal(state)
	c := checkpointTestClient(t, path, 4, 55)
	c.name = "Bot"
	c.framePaced, c.testCombatBarrier = true, true
	c.checkpointMode = "combat"
	c.checkpointControl = filepath.Join(t.TempDir(), "control")
	c.checkpointRestore = &checkpoint.Capture{CaptureRequest: checkpoint.CaptureRequest{Version: 1, ID: "root", Boundary: checkpoint.CombatPreDecisionBoundary, Map: "base1", Frame: 175, Generation: 3}, SelfEntity: 1, Combat: raw, Health: 100}
	fresh := checkpointTestSnapshot(1, 5)
	fresh.OnGround = false
	c.latestFrame = 1
	c.planner.World.Snapshot = fresh
	if err := c.restoreCheckpointSnapshot(fresh); err != nil {
		t.Fatal(err)
	}
	if !c.checkpointRestored || !c.testCombatGo || !c.testTeleportSent || !c.testHandGrenadeArmDone || c.lastMoveFrame != 0 || c.combatControl.last.Frame != 0 || c.combatControl.observation != nil {
		t.Fatal("invalid restore boundary/setup state")
	}
	if held, err := c.holdCombatCheckpoint(); err != nil || !held {
		t.Fatal("load root not held", held, err)
	}
	want := *c.checkpointHeld
	if err := checkpoint.WriteCapture(c.checkpointControl+".released.json", checkpoint.SourceRelease{CaptureRequest: want, State: "native_released"}); err != nil {
		t.Fatal(err)
	}
	if held, err := c.holdCombatCheckpoint(); err != nil || held {
		t.Fatal("load release failed", held, err)
	}
	c.planner.World.Updated = time.Now()
	a := c.combatObservation(time.Now())
	if a.Identity.Frame != 1 || a.Identity.Life != 1 || len(a.History) != 4 || a.History[3].Identity.Frame != 0 {
		t.Fatal("prior memory lost on first native root")
	}
	if _, err := c.combatControl.provider.(*policy.HJEPA).Decide(a); err != nil {
		t.Fatal(err)
	}
	if err := c.restoreCheckpointSnapshot(fresh); err != nil {
		t.Fatal(err)
	}
}
