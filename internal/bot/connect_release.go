package bot

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// Operational startup gate: provider validation/loading has completed, but no
// UDP packets have been sent. This never pauses or rewrites gameplay frames.
func waitConnectRelease(ctx context.Context, cfg Config) error {
	if cfg.ConnectReadyFile == "" && cfg.ConnectReleaseFile == "" {
		return nil
	}
	if cfg.Host != "127.0.0.1" || cfg.ConnectReadyFile == "" || cfg.ConnectReleaseFile == "" || strings.EqualFold(filepath.Clean(cfg.ConnectReadyFile), filepath.Clean(cfg.ConnectReleaseFile)) {
		return fmt.Errorf("connect gate requires loopback and distinct ready/release paths")
	}
	if _, err := os.Stat(cfg.ConnectReleaseFile); !os.IsNotExist(err) {
		return fmt.Errorf("connect release must be absent at startup")
	}
	nonce := make([]byte, 16)
	if _, err := rand.Read(nonce); err != nil {
		return err
	}
	token := hex.EncodeToString(nonce)
	data, _ := json.Marshal(struct {
		PID   int    `json:"pid"`
		Token string `json:"token"`
	}{os.Getpid(), token})
	f, err := os.OpenFile(cfg.ConnectReadyFile, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
	if err != nil {
		return fmt.Errorf("create connect ready: %w", err)
	}
	_, writeErr := f.Write(data)
	closeErr := f.Close()
	if writeErr != nil {
		return writeErr
	}
	if closeErr != nil {
		return closeErr
	}
	deadline := time.NewTimer(30 * time.Second)
	defer deadline.Stop()
	poll := time.NewTicker(10 * time.Millisecond)
	defer poll.Stop()
	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-deadline.C:
			return fmt.Errorf("connect release timeout")
		case <-poll.C:
			contents, err := os.ReadFile(cfg.ConnectReleaseFile)
			if os.IsNotExist(err) {
				continue
			}
			if err != nil {
				return err
			}
			if strings.TrimSpace(string(contents)) == token {
				return nil
			}
			// An incomplete atomic publication is not approval to connect.
		}
	}
}
