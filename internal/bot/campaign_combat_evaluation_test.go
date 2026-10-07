package bot

import "testing"

func TestCampaignCombatEvaluationIsolation(t *testing.T) {
	base := Config{Host: "127.0.0.1", Campaign: true, FramePaced: true, CombatCapture: true, TracePath: "trace", CombatMode: "learned", CombatProviderFile: "weights", GameFrames: 1800}
	if err := validateCampaignCombatEvaluation(base); err != nil {
		t.Fatal(err)
	}
	for _, mutate := range []func(*Config){
		func(c *Config) { c.Host = "192.168.1.1" },
		func(c *Config) { c.Campaign = false },
		func(c *Config) { c.CombatCapture = false },
		func(c *Config) { c.TestTeleport = "0,0,0" },
		func(c *Config) { c.TestInvulnerable = true },
		func(c *Config) { c.TestWeaponSwitchFixture = "parasite_weapons" },
		func(c *Config) { c.CombatProviderFile = "" },
		func(c *Config) { c.System1Model = "llm" },
		func(c *Config) { c.GameFrames = 0 },
		func(c *Config) { c.TestDisableSearch = true },
		func(c *Config) { c.TestDoorProbe = true },
	} {
		c := base
		mutate(&c)
		if validateCampaignCombatEvaluation(c) == nil {
			t.Fatalf("invalid configuration accepted: %+v", c)
		}
	}
}
