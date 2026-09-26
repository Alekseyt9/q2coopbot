package bot

import (
	"encoding/json"
	"fmt"
	"os"
	"q2coopbot/internal/harness"
)

type scenarioCompletion struct {
	Map        string `json:"map"`
	Generation int    `json:"generation"`
	EndFrame   int    `json:"end_frame"`
	State      string `json:"state"`
}

func (c *Client) publishScenarioCompletion() error {
	if c.scenario == nil || c.scenarioResultPath == "" || c.scenarioResultSent {
		return nil
	}
	s := c.scenario.Status
	if s.State != "completed" && s.State != "failed" {
		return nil
	}
	data, err := json.Marshal(scenarioCompletion{Map: c.planner.World.Map, Generation: c.spawncount, EndFrame: s.EndFrame, State: s.State})
	if err != nil {
		return err
	}
	tmp := c.scenarioResultPath + ".tmp"
	if err = os.WriteFile(tmp, data, 0600); err != nil {
		return err
	}
	if err = os.Rename(tmp, c.scenarioResultPath); err != nil {
		return err
	}
	c.scenarioResultSent = true
	return nil
}

func (c *Client) scenarioShouldStop() (bool, error) {
	if c.scenarioCompletion == nil {
		data, err := os.ReadFile(c.scenarioResultPath)
		if os.IsNotExist(err) {
			return false, nil
		}
		if err != nil {
			return false, err
		}
		var end scenarioCompletion
		if err = json.Unmarshal(data, &end); err != nil {
			return false, err
		}
		if end.EndFrame < 1 || end.EndFrame > 10000 || end.State != "completed" && end.State != "failed" {
			return false, fmt.Errorf("invalid scenario completion")
		}
		c.scenarioCompletion = &end
	}
	end := c.scenarioCompletion
	if end.Map != c.planner.World.Map || end.Generation != c.spawncount {
		return false, fmt.Errorf("scenario completion belongs to another map generation")
	}
	// Wait for the next server snapshot: it acknowledges the final command.
	return c.lastMoveFrame >= end.EndFrame+c.scenarioTailFrames && c.latestFrame > c.lastMoveFrame, nil
}

func (c *Client) scenarioStatus() *harness.Status {
	if c.scenario == nil {
		return nil
	}
	status := c.scenario.Status
	return &status
}
