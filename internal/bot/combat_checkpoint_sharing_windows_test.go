package bot

import (
	"path/filepath"
	"syscall"
	"testing"
	"time"

	"q2coopbot/internal/harness/checkpoint"
)

// Reproduce the actual Win32 sharing failure seen while a control file was
// published, rather than injecting an arbitrary error into the capture code.
func exclusiveCheckpointFile(t *testing.T, path string) func() {
	t.Helper()
	name, err := syscall.UTF16PtrFromString(path)
	if err != nil {
		t.Fatal(err)
	}
	h, err := syscall.CreateFile(name, syscall.GENERIC_READ, 0, nil, syscall.OPEN_EXISTING, syscall.FILE_ATTRIBUTE_NORMAL, 0)
	if err != nil {
		t.Fatal(err)
	}
	release := func() {
		if h != syscall.InvalidHandle {
			_ = syscall.CloseHandle(h)
			h = syscall.InvalidHandle
		}
	}
	t.Cleanup(release)
	return release
}

func TestCombatCheckpointWaitsThroughRealWindowsControlLocks(t *testing.T) {
	c := checkpointTestClient(t, combatCheckpointModel(t), 2, 3)
	c.framePaced, c.testCombatBarrier = true, true
	c.name = "Bot"
	c.checkpointControl = filepath.Join(t.TempDir(), "control")
	for step := 0; step < 5; step++ {
		c.planner.World.Snapshot = checkpointTestSnapshot(170+step, step)
		c.planner.World.Updated = time.Now()
		o := c.combatObservation(time.Now())
		c.combatControl.engagement.observe(o, nil)
	}
	r := checkpoint.CaptureRequest{Version: 1, ID: "locked", Map: "base1", Frame: 175, Generation: 3, Boundary: checkpoint.CombatPreDecisionBoundary}
	c.latestFrame, c.lastMoveFrame = 175, 174
	c.planner.World.Snapshot = checkpointTestSnapshot(175, 5)
	if err := checkpoint.WriteCapture(c.checkpointControl, r); err != nil {
		t.Fatal(err)
	}
	unlock := exclusiveCheckpointFile(t, c.checkpointControl)
	if err := c.captureCheckpointRequest(); !checkpoint.ControlSharingError(err) {
		t.Fatalf("not a retryable sharing error: %v", err)
	}
	if c.combatControl.last.Frame != 174 || c.checkpointCapturedID != "" || c.lastMoveFrame != 174 {
		t.Fatal("locked request consumed root")
	}
	unlock()
	if err := c.captureCheckpointRequest(); err != nil {
		t.Fatal(err)
	}
	path := c.checkpointControl + ".released.json"
	if err := checkpoint.WriteCapture(path, checkpoint.SourceRelease{CaptureRequest: r, State: "native_released"}); err != nil {
		t.Fatal(err)
	}
	unlock = exclusiveCheckpointFile(t, path)
	if held, err := c.holdCombatCheckpoint(); err != nil || !held {
		t.Fatal("locked release must keep waiting", held, err)
	}
	unlock()
	if held, err := c.holdCombatCheckpoint(); err != nil || held {
		t.Fatal("release not accepted after unlock", held, err)
	}
}
