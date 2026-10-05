package bot

import "q2coopbot/internal/policy"

func (c *Client) beginTestCombat() {
	if c.testSynchronous {
		c.combatControl.history = policy.History{}
		// The next command starts the fixed game-frame budget. Preparation
		// duration must not reduce the controlled combat/evaluation horizon.
		c.firstMoveFrame = -1
	}
	c.testCombatGo = true
	c.testCombatGoFrame = c.latestFrame
}
