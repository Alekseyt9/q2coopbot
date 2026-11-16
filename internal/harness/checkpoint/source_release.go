package checkpoint

import (
	"context"
	"fmt"
	"strings"
	"time"
)

// SourceRelease is published only after the native save barrier is confirmed
// clear. The client waits for this exact identity before sending its root cmd.
type SourceRelease struct {
	CaptureRequest
	State string `json:"state"`
}

func publishSourceRelease(ctx context.Context, c Config, password string) error {
	return publishBarrierRelease(ctx, c, password, c.Barrier)
}

func publishBarrierRelease(ctx context.Context, c Config, password string, b *Barrier) error {
	if b == nil || b.Boundary != CombatPreDecisionBoundary {
		return nil
	}
	reply, err := request(ctx, c.Server, password, "sv_test_checkpoint_frame", time.Duration(c.TimeoutMS)*time.Millisecond)
	if err != nil {
		return err
	}
	if !strings.Contains(reply, `"sv_test_checkpoint_frame" is "0"`) {
		return fmt.Errorf("source native barrier release not confirmed")
	}
	r := SourceRelease{CaptureRequest: CaptureRequest{Version: 1, ID: b.ID, Map: b.Map, Frame: b.Frame, Generation: b.Generation, Boundary: b.Boundary}, State: "native_released"}
	for _, control := range b.Controls {
		if err := WriteCapture(control+".released.json", r); err != nil {
			return err
		}
	}
	return nil
}

// SourceReleaseReady ignores receipts for older captures, but rejects a receipt
// that claims this ID with a different world/command boundary.
func SourceReleaseReady(path string, want CaptureRequest) (bool, error) {
	var r SourceRelease
	if err := readJSON(path, &r); err != nil {
		return false, err
	}
	if r.ID != want.ID {
		return false, nil
	}
	if r.CaptureRequest != want || r.State != "native_released" || want.Boundary != CombatPreDecisionBoundary {
		return false, fmt.Errorf("source release identity/state mismatch")
	}
	return true, nil
}
