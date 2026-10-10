package bot

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestConnectGateRequiresCurrentReadyIdentity(t *testing.T) {
	dir := t.TempDir()
	cfg := Config{Host: "127.0.0.1", ConnectReadyFile: filepath.Join(dir, "ready.json"), ConnectReleaseFile: filepath.Join(dir, "release.txt")}
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	done := make(chan error, 1)
	go func() { done <- waitConnectRelease(ctx, cfg) }()
	var ready struct {
		PID   int    `json:"pid"`
		Token string `json:"token"`
	}
	for ready.Token == "" {
		select {
		case err := <-done:
			t.Fatalf("gate exited early: %v", err)
		default:
		}
		data, _ := os.ReadFile(cfg.ConnectReadyFile)
		_ = json.Unmarshal(data, &ready)
		if ctx.Err() != nil {
			t.Fatal(ctx.Err())
		}
		time.Sleep(time.Millisecond)
	}
	if ready.PID != os.Getpid() {
		t.Fatal("wrong ready PID")
	}
	if err := os.WriteFile(cfg.ConnectReleaseFile, []byte("stale-token"), 0600); err != nil {
		t.Fatal(err)
	}
	select {
	case err := <-done:
		t.Fatalf("stale release passed: %v", err)
	case <-time.After(30 * time.Millisecond):
	}
	if err := os.WriteFile(cfg.ConnectReleaseFile, []byte(ready.Token), 0600); err != nil {
		t.Fatal(err)
	}
	if err := <-done; err != nil {
		t.Fatal(err)
	}
	if err := waitConnectRelease(ctx, cfg); err == nil {
		t.Fatal("stale paths reused")
	}
}

func TestConnectGateCancellationAndIsolation(t *testing.T) {
	if err := waitConnectRelease(context.Background(), Config{}); err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	cfg := Config{Host: "127.0.0.1", ConnectReadyFile: filepath.Join(dir, "ready"), ConnectReleaseFile: filepath.Join(dir, "release")}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := waitConnectRelease(ctx, cfg); err != context.Canceled {
		t.Fatal(err)
	}
	if _, err := os.Stat(cfg.ConnectReleaseFile); !os.IsNotExist(err) {
		t.Fatal("cancellation created release")
	}
	cfg.Host = "0.0.0.0"
	if err := waitConnectRelease(context.Background(), cfg); err == nil {
		t.Fatal("non-loopback gate allowed")
	}
	cfg.Host = "127.0.0.1"
	cfg.ConnectReleaseFile = cfg.ConnectReadyFile
	if err := waitConnectRelease(context.Background(), cfg); err == nil {
		t.Fatal("shared ready/release path allowed")
	}
}
