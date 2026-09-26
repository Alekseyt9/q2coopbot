package bot

import (
	"fmt"
	"q2coopbot/internal/harness"
)

// This is a test fixture. It does not add automatic respawn to the bot policy.
type observerRespawnState struct {
	killed     bool
	deathFrame int
	done       bool
}

func (s *observerRespawnState) tick(f *harness.ObserverRespawn, start, frame int, health int16) (kill, respawn bool, err error) {
	if f == nil || start == 0 || s.done {
		return
	}
	target := start + f.AfterFrames
	if !s.killed {
		if frame < target {
			return
		}
		if frame != target || health <= 0 {
			return false, false, fmt.Errorf("observer respawn fixture missed living trigger")
		}
		s.killed = true
		return true, false, nil
	}
	if health <= 0 {
		if s.deathFrame == 0 {
			s.deathFrame = frame
		}
		return false, (frame-s.deathFrame)%2 == 0, nil
	}
	if s.deathFrame > 0 {
		s.done = true
	}
	return
}
