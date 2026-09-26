package bot

import "q2coopbot/internal/quake"

// Harness-only perception fault. Server items and player state are unchanged.
func (c *Client) maskTestHealth(s quake.Snapshot) quake.Snapshot {
	if !c.testHealthMasked(s) {
		return s
	}
	items := make([]quake.Object, 0, len(s.Pickups))
	for _, item := range s.Pickups {
		if item.Class != "item_health" {
			items = append(items, item)
		}
	}
	s.Pickups = items
	return s
}

func (c *Client) testHealthMasked(s quake.Snapshot) bool {
	if len(c.testHideHealthFrames) != 2 || !c.testTeleportSent || s.Map != c.testTeleportMap {
		return false
	}
	age := c.testScenarioAge(s.Frame)
	if age < c.testHideHealthFrames[0] || age >= c.testHideHealthFrames[1] {
		return false
	}
	return true
}
