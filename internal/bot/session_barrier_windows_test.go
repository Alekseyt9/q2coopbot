package bot

import (
	"errors"
	"os"
	"path/filepath"
	"q2coopbot/internal/harness"
	"syscall"
	"testing"
	"time"
)

func TestBarrierRetriesWindowsSharingViolation(t *testing.T) {
	dir := t.TempDir()
	completion := filepath.Join(dir, "end.json")
	barrier := completion + ".barrier"
	if err := os.Mkdir(barrier, 0700); err != nil {
		t.Fatal(err)
	}
	actorPath := filepath.Join(barrier, "0-7-actor.json")
	for _, role := range []string{"actor", "observer"} {
		if err := writeSessionSignal(filepath.Join(barrier, "0-7-"+role+".json"), sessionReady{Map: "base1", Generation: 7, Phase: 0, Frame: 40, Role: role}); err != nil {
			t.Fatal(err)
		}
	}
	ptr, err := syscall.UTF16PtrFromString(actorPath)
	if err != nil {
		t.Fatal(err)
	}
	handle, err := syscall.CreateFile(ptr, syscall.GENERIC_READ, 0, nil, syscall.OPEN_EXISTING, syscall.FILE_ATTRIBUTE_NORMAL, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer syscall.CloseHandle(handle)
	if _, err := readSessionReady(actorPath); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("sharing violation not retryable: %v", err)
	}
	c := Client{sessionDefinition: &harness.Session{ReadinessBarrier: true, TransitionTimeoutMS: 1000}, begun: true, frameReady: true, sessionMap: "base1", spawncount: 7, sessionReadySent: true, scenarioResultPath: completion, latestFrame: 41, sessionSetupAt: time.Now()}
	if err := c.sessionBarrier(time.Now()); err != nil || c.sessionStartFrame != 0 {
		t.Fatal("locked peer signal must remain pending", err)
	}
}

func TestCompletionRetriesWindowsSharingViolation(t *testing.T) {
	path := filepath.Join(t.TempDir(), "completion.json")
	if err := os.WriteFile(path, []byte(`{"map":"base2","generation":7,"end_frame":40,"state":"completed"}`), 0600); err != nil {
		t.Fatal(err)
	}
	ptr, err := syscall.UTF16PtrFromString(path)
	if err != nil {
		t.Fatal(err)
	}
	handle, err := syscall.CreateFile(ptr, syscall.GENERIC_READ, 0, nil, syscall.OPEN_EXISTING, syscall.FILE_ATTRIBUTE_NORMAL, 0)
	if err != nil {
		t.Fatal(err)
	}
	locked := true
	defer func() {
		if locked {
			syscall.CloseHandle(handle)
		}
	}()
	c := Client{scenarioResultPath: path, spawncount: 7, lastMoveFrame: 45, latestFrame: 46, scenarioTailFrames: 5, planner: &Planner{}}
	c.planner.World.Map = "base2"
	if stop, err := c.scenarioShouldStop(); stop || err != nil || c.scenarioCompletion != nil {
		t.Fatal("locked completion must remain pending", stop, err)
	}
	if err := syscall.CloseHandle(handle); err != nil {
		t.Fatal(err)
	}
	locked = false
	if stop, err := c.scenarioShouldStop(); !stop || err != nil {
		t.Fatal("completion was not acknowledged after unlock", stop, err)
	}
}
