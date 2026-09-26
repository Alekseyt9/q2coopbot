package bot

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"q2coopbot/internal/harness/coord"
	"time"
)

type sessionReady = coord.Ready

func (c *Client) sessionBarrier(now time.Time) error {
	if c.sessionDefinition == nil || !c.sessionDefinition.ReadinessBarrier || !c.begun || !c.frameReady || c.sessionMap == "" || c.sessionStartFrame != 0 {
		return nil
	}
	if now.Sub(c.sessionSetupAt) >= time.Duration(c.sessionDefinition.TransitionTimeoutMS)*time.Millisecond {
		return fmt.Errorf("session readiness timeout at phase %d", c.sessionPhase)
	}
	role, other := "observer", "actor"
	if c.session != nil {
		role, other = other, role
	}
	dir := c.scenarioResultPath + ".barrier"
	path := func(role string) string {
		return filepath.Join(dir, fmt.Sprintf("%d-%d-%s.json", c.sessionPhase, c.spawncount, role))
	}
	if !c.sessionReadySent {
		s := c.planner.World.Snapshot
		p := c.testTeleportPosition
		// The placement target can be above the floor; require observed XY arrival,
		// bounded vertical settling, a living player and ground contact.
		if !c.testTeleportSent || c.latestFrame <= c.testTeleportSentFrame || !s.OnGround || s.Health <= 0 || math.Hypot(s.Self[0]-p[0], s.Self[1]-p[1]) > 16 || math.Abs(s.Self[2]-p[2]) > 32 {
			return nil
		}
		if err := os.MkdirAll(dir, 0700); err != nil {
			return err
		}
		if _, err := os.Stat(path(role)); err == nil {
			return fmt.Errorf("stale session readiness file: use a fresh run directory")
		} else if !errors.Is(err, os.ErrNotExist) {
			return err
		}
		data, _ := json.Marshal(sessionReady{Map: c.sessionMap, Generation: c.spawncount, Phase: c.sessionPhase, Frame: c.latestFrame, Role: role})
		tmp := path(role) + ".tmp"
		if err := os.WriteFile(tmp, data, 0600); err != nil {
			return err
		}
		if err := os.Rename(tmp, path(role)); err != nil {
			return err
		}
		c.sessionReadySent = true
	}
	a, err := readSessionReady(path(role))
	if errors.Is(err, os.ErrNotExist) {
		return nil
	}
	if err != nil {
		return err
	}
	b, err := readSessionReady(path(other))
	if errors.Is(err, os.ErrNotExist) {
		return nil
	}
	if err != nil {
		return err
	}
	start, err := sessionBarrierStart(a, b, c.sessionMap, c.spawncount, c.sessionPhase)
	if err != nil {
		return err
	}
	if c.latestFrame >= start {
		return fmt.Errorf("session readiness start missed: current %d start %d", c.latestFrame, start)
	}
	c.sessionStartFrame = start
	return nil
}

func readSessionReady(path string) (sessionReady, error) { return coord.Read(path) }

func sessionBarrierStart(a, b sessionReady, mapName string, generation, phase int) (int, error) {
	return coord.Start(a, b, mapName, generation, phase)
}
