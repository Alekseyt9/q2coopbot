package bot

import "fmt"

// Real-time two-player geometry acceptance is distinct from single-player
// lockstep learning. Only the bounded local diagnostic provider is allowed.
func validateHitscanGuardProbe(c Config) error {
	if c.Host != "127.0.0.1" || c.CombatMode != "learned" || !c.FramePaced || !c.CombatCapture || c.TracePath == "" || c.CombatProviderFile == "" || c.TestTeleport == "" || c.TestTeleportMap != "base1" || c.TestWeaponSwitchFixture != "parasite_machinegun" || c.TestSynchronous || c.TestCombatBarrier || c.CombatLiveCompanion || c.TestCampaignCombatEvaluation || c.Idle || c.TestScenario != "" || c.TestSession != "" || c.System1Model != "" || c.System2Model != "" {
		return fmt.Errorf("hitscan guard diagnostic requires loopback real-time fixed Machinegun placement, capture and no neural/learning/LLM overrides")
	}
	return nil
}
