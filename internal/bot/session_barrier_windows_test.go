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
