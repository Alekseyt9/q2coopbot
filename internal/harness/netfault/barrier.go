package netfault

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"q2coopbot/internal/harness/coord"
)

type BarrierEvidence struct {
	Actor    coord.Ready `json:"actor"`
	Observer coord.Ready `json:"observer"`
}

func barrierAt(c Config, f GameFrame) (*BarrierEvidence, error) {
	path := func(role string) string {
		return filepath.Join(c.ArmBarrierDir, fmt.Sprintf("%d-%d-%s.json", c.ArmPhase, f.Generation, role))
	}
	a, err := coord.Read(path("actor"))
	if errors.Is(err, os.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	b, err := coord.Read(path("observer"))
	if errors.Is(err, os.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	start, err := coord.Start(a, b, f.Map, f.Generation, c.ArmPhase)
	if err != nil {
		return nil, err
	}
	if f.Frame < start {
		return nil, nil
	}
	if f.Frame > start {
		return nil, fmt.Errorf("network barrier start missed: %d > %d", f.Frame, start)
	}
	return &BarrierEvidence{Actor: a, Observer: b}, nil
}

func verifyBarrier(c Config, e Event) error {
	if e.Barrier == nil {
		return fmt.Errorf("missing readiness evidence")
	}
	for _, f := range e.Frames {
		start, err := coord.Start(e.Barrier.Actor, e.Barrier.Observer, f.Map, f.Generation, c.ArmPhase)
		if err == nil && f.Frame == start && e.Barrier.Actor.Role == "actor" && e.Barrier.Observer.Role == "observer" {
			return nil
		}
	}
	return fmt.Errorf("readiness evidence does not match trigger frame")
}
