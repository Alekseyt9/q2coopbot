package bot

import "testing"

func TestLiveLearnedCompanionNaturalGameplay(t *testing.T) {
	base := Config{Host: "127.0.0.1", FramePaced: true, CombatCapture: true, TracePath: "trace", CombatMode: "learned", CombatProviderFile: "weights", CombatLiveCompanion: true}
	if err := validateLiveLearnedCompanion(base); err != nil {
		t.Fatal(err)
	}
	for _, mutate := range []func(*Config){
		func(c *Config) { c.Campaign = true },
		func(c *Config) { c.TestCampaignCombatEvaluation = true },
		func(c *Config) { c.TestTeleport = "0,0,0" },
		func(c *Config) { c.TestInvulnerable = true },
		func(c *Config) { c.Host = "192.168.1.1" },
		func(c *Config) { c.FramePaced = false },
		func(c *Config) { c.CombatProviderFile = "" },
		func(c *Config) { c.CombatMode = "rules" },
	} {
		c := base
		mutate(&c)
		if validateLiveLearnedCompanion(c) == nil {
			t.Fatalf("invalid live config accepted: %+v", c)
		}
	}
}
