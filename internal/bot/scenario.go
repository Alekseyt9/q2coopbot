package bot

import "q2coopbot/internal/harness"

func (c *Client) scenarioStatus() *harness.Status {
	if c.scenario == nil {
		return nil
	}
	status := c.scenario.Status
	return &status
}
