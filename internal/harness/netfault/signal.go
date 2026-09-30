package netfault

import (
	"errors"
	"fmt"
	"os"
	"q2coopbot/internal/harness/coord"
)

// The orchestrator arms loss only after observing the intended gameplay state.
// Record both the observed frame and the actual triggering network frame.
func signalAt(path string, f GameFrame) (*coord.Ready, error) {
	r, err := coord.Read(path)
	if errors.Is(err, os.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	if r.Map != f.Map || r.Generation != f.Generation || r.Role != "observer" || r.Phase != 0 || r.Frame < 1 || r.Frame > 100000 {
		return nil, fmt.Errorf("network state signal identity mismatch")
	}
	if f.Frame < r.Frame {
		return nil, nil
	}
	if f.Frame-r.Frame > 20 {
		return nil, fmt.Errorf("network state signal stale")
	}
	return &r, nil
}
