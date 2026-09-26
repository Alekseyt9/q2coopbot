package bot

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

func writeSessionSignal(path string, r sessionReady) error {
	if _, err := os.Stat(path); err == nil {
		return fmt.Errorf("stale session transition signal")
	} else if !os.IsNotExist(err) {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0700); err != nil {
		return err
	}
	data, _ := json.Marshal(r)
	if err := os.WriteFile(path+".tmp", data, 0600); err != nil {
		return err
	}
	return os.Rename(path+".tmp", path)
}

// Stop both streams at the boundary and acknowledge their last usercmd before
// loading a new generation. The observer must include the actor's terminal frame.
func (c *Client) sessionTransitionBarrier() (pause, ready bool, err error) {
	if c.sessionDefinition == nil || !c.frameReady || c.sessionMap == "" {
		return false, false, nil
	}
	dir := c.scenarioResultPath + ".barrier"
	path := func(role string) string {
		return filepath.Join(dir, fmt.Sprintf("%d-%d-transition-%s.json", c.sessionPhase, c.spawncount, role))
	}
	valid := func(r sessionReady, role string) bool {
		return r.Map == c.sessionMap && r.Generation == c.spawncount && r.Phase == c.sessionPhase && r.Role == role && r.Frame > 0
	}
	if c.session != nil {
		if c.session.Status.State != "waiting_map" || c.session.Status.PhaseIndex != c.sessionPhase {
			return false, false, nil
		}
		if c.sessionPendingMap == "" {
			return true, false, nil
		}
		if !c.sessionTransitionRequested {
			err := writeSessionSignal(path("actor"), sessionReady{Map: c.sessionMap, Generation: c.spawncount, Phase: c.sessionPhase, Frame: c.lastMoveFrame, Role: "actor"})
			if err != nil {
				return true, false, err
			}
			c.sessionTransitionRequested = true
		}
		ack, err := readSessionReady(path("observer"))
		if os.IsNotExist(err) {
			return true, false, nil
		}
		if err != nil {
			return true, false, err
		}
		if !valid(ack, "observer") || ack.Frame < c.lastMoveFrame {
			return true, false, fmt.Errorf("invalid transition acknowledgement")
		}
		return true, c.latestFrame > c.lastMoveFrame, nil
	}
	request, err := readSessionReady(path("actor"))
	if os.IsNotExist(err) {
		return false, false, nil
	}
	if err != nil {
		return true, false, err
	}
	if !valid(request, "actor") {
		return true, false, fmt.Errorf("invalid transition request")
	}
	if c.lastMoveFrame < request.Frame {
		return false, false, nil
	}
	if !c.sessionTransitionAcked && c.latestFrame > c.lastMoveFrame {
		err := writeSessionSignal(path("observer"), sessionReady{Map: c.sessionMap, Generation: c.spawncount, Phase: c.sessionPhase, Frame: c.lastMoveFrame, Role: "observer"})
		if err != nil {
			return true, false, err
		}
		c.sessionTransitionAcked = true
	}
	if c.sessionTransitionAcked && c.sessionPhase+1 < len(c.sessionDefinition.Phases) && c.sessionDefinition.Phases[c.sessionPhase+1].Entry == "reconnect" {
		release, err := readSessionReady(path("release"))
		if os.IsNotExist(err) {
			return true, false, nil
		}
		if err != nil {
			return true, false, err
		}
		if !valid(release, "release") {
			return true, false, fmt.Errorf("invalid reconnect release")
		}
		return true, false, c.reconnect()
	}
	return true, false, nil
}

func (c *Client) releaseSessionReconnect() error {
	path := filepath.Join(c.scenarioResultPath+".barrier", fmt.Sprintf("%d-%d-transition-release.json", c.sessionPhase, c.spawncount))
	return writeSessionSignal(path, sessionReady{Map: c.sessionMap, Generation: c.spawncount, Phase: c.sessionPhase, Frame: c.lastMoveFrame, Role: "release"})
}
