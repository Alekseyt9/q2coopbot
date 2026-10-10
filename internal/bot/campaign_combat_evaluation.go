package bot

import (
	"fmt"
	"reflect"
	"strings"
)

// This is a local evaluation opt-in, independent of the isolated training fixtures.
// Navigation and explicitly logged fallbacks remain under the campaign planner.
func validateCampaignCombatEvaluation(c Config) error {
	if !c.Campaign || c.GameFrames <= 0 {
		return fmt.Errorf("campaign combat evaluation requires campaign mode and a finite game frame budget")
	}
	return validateNaturalCombat(c)
}

// Explicit human-session opt-in keeps the ordinary companion planner for
// following the teammate; learned combat alone owns active engagements.
func validateLiveLearnedCompanion(c Config) error {
	if c.Campaign || c.TestCampaignCombatEvaluation || c.CombatMode != "learned" {
		return fmt.Errorf("live learned companion requires companion mode and direct learned combat")
	}
	return validateNaturalCombat(c)
}

func validateNaturalCombat(c Config) error {
	fields := reflect.ValueOf(c)
	for i := 0; i < fields.NumField(); i++ {
		name := fields.Type().Field(i).Name
		if !strings.HasPrefix(name, "Test") || name == "TestCampaignCombatEvaluation" || name == "TestRCONPassword" {
			continue
		}
		// The config loader supplies this inert default even without change_map.
		if name == "TestChangeAfter" && (c.TestChangeAfter == 0 || c.TestChangeAfter == 20) {
			continue
		}
		if !fields.Field(i).IsZero() {
			return fmt.Errorf("campaign combat evaluation forbids fixture override %s", name)
		}
	}
	if c.Host != "127.0.0.1" || !c.FramePaced || !c.CombatCapture || c.TracePath == "" || c.System1Model != "" || c.System2Model != "" || c.Idle || c.TestSynchronous || c.TestCombatBarrier || c.TestTeleport != "" || c.TestTeleportAfter != "" || c.TestTeleportReturn != "" || c.TestSpawnSoldier != "" || c.TestSpawnClass != "" || c.TestCampaignGoal != "" || c.TestWalkTarget != "" || c.TestScenario != "" || c.TestSession != "" || c.TestHoldPosition || c.TestHoldPositionMap != "" || c.TestInvulnerable || c.TestInitialHealth != 0 || c.TestWeaponSwitchFixture != "" || c.TestChangeMap != "" || c.TestLight != nil || c.TestLineCross || c.CheckpointRestore != "" || c.CheckpointControl != "" {
		return fmt.Errorf("campaign combat evaluation requires an isolated loopback campaign, capture, frame pacing and natural gameplay without fixture overrides or LLMs")
	}
	if c.CombatMode != "rules" && (c.CombatMode != "learned" || c.CombatProviderFile == "") {
		return fmt.Errorf("campaign combat evaluation requires rules or direct learned with a provider")
	}
	return nil
}
