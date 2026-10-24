package bot

import (
	"context"
	"q2coopbot/internal/policy"
	"strings"
	"testing"
)

func TestCombatExecutionContractRejectsUnsafeSessions(t *testing.T) {
	base := Config{CombatExecutionContract: policy.CheckedGroundCombatExecution, Host: "127.0.0.1", CombatMode: "learned",
		CombatCapture: true, TestSynchronous: true, TestCombatOnly: true, CombatCheckedWallSlide: true}
	for _, mutate := range []func(*Config){
		func(c *Config) { c.CombatExecutionContract = "unknown" },
		func(c *Config) { c.Host = "192.0.2.1" },
		func(c *Config) { c.CombatMode = "rules" },
		func(c *Config) { c.CombatCapture = false },
		func(c *Config) { c.TestSynchronous = false },
		func(c *Config) { c.TestCombatOnly = false },
		func(c *Config) { c.CombatCheckedWallSlide = false },
		func(c *Config) { c.CombatActionQualityGuard = true },
		func(c *Config) { c.TestStoppedCrouchProbeAllFrames = true },
		func(c *Config) { c.TestCheckedGroundRequestProbeAllFrames = true },
	} {
		c := base
		mutate(&c)
		if err := Run(context.Background(), c); err == nil || !strings.Contains(err.Error(), "execution contract") {
			t.Fatal("invalid contract reached runtime", err)
		}
	}
}
