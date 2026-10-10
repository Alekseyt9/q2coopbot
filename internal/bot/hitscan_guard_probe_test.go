package bot

import "testing"

func TestHitscanGuardProbeIsolation(t *testing.T) {
	base := Config{Host: "127.0.0.1", CombatMode: "learned", FramePaced: true, CombatCapture: true, TracePath: "trace", CombatProviderFile: "probe", TestTeleport: "32,-224,24.125", TestTeleportMap: "base1", TestWeaponSwitchFixture: "parasite_machinegun"}
	if err := validateHitscanGuardProbe(base); err != nil {
		t.Fatal(err)
	}
	for _, change := range []func(*Config){
		func(c *Config) { c.Host = "0.0.0.0" }, func(c *Config) { c.TestSynchronous = true }, func(c *Config) { c.TestCombatBarrier = true },
		func(c *Config) { c.CombatLiveCompanion = true }, func(c *Config) { c.TestCampaignCombatEvaluation = true },
		func(c *Config) { c.TestWeaponSwitchFixture = "parasite_weapons" }, func(c *Config) { c.System1Model = "model" },
	} {
		c := base
		change(&c)
		if validateHitscanGuardProbe(c) == nil {
			t.Fatal("invalid diagnostic config accepted", c)
		}
	}
}
