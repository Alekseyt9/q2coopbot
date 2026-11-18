package checkpoint

import (
	"encoding/json"
	"os"
	"path/filepath"
	"syscall"
	"testing"
	"time"
)

func TestCapturePublicationWaitsForWindowsReader(t *testing.T) {
	path := filepath.Join(t.TempDir(), "control.json")
	if err := WriteCapture(path, map[string]int{"frame": 1}); err != nil {
		t.Fatal(err)
	}
	name, err := syscall.UTF16PtrFromString(path)
	if err != nil {
		t.Fatal(err)
	}
	handle, err := syscall.CreateFile(name, syscall.GENERIC_READ, syscall.FILE_SHARE_READ, nil, syscall.OPEN_EXISTING, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if handle != syscall.InvalidHandle {
			syscall.CloseHandle(handle)
		}
	}()
	done := make(chan error, 1)
	go func() { done <- WriteCapture(path, map[string]int{"frame": 2}) }()
	select {
	case err := <-done:
		t.Fatalf("publication returned while reader held target: %v", err)
	case <-time.After(60 * time.Millisecond):
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var before map[string]int
	if json.Unmarshal(data, &before) != nil || before["frame"] != 1 {
		t.Fatal("partial publication", string(data))
	}
	if err := syscall.CloseHandle(handle); err != nil {
		t.Fatal(err)
	}
	handle = syscall.InvalidHandle
	select {
	case err := <-done:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("publication did not recover")
	}
	data, err = os.ReadFile(path)
	var after map[string]int
	if err != nil || json.Unmarshal(data, &after) != nil || after["frame"] != 2 {
		t.Fatal("new publication missing", err, string(data))
	}
}

func TestCapturePublicationFailureKeepsTargetAndCleansPending(t *testing.T) {
	root := t.TempDir()
	path := filepath.Join(root, "directory")
	if err := os.Mkdir(path, 0700); err != nil {
		t.Fatal(err)
	}
	start := time.Now()
	if err := WriteCapture(path, map[string]int{"frame": 2}); err == nil {
		t.Fatal("replaced directory with capture")
	}
	if time.Since(start) > 2*time.Second {
		t.Fatal("unbounded publication retry")
	}
	info, err := os.Stat(path)
	if err != nil || !info.IsDir() {
		t.Fatal("target damaged", err)
	}
	entries, err := os.ReadDir(root)
	if err != nil || len(entries) != 1 {
		t.Fatal("failed publication left pending files", err, entries)
	}
}
