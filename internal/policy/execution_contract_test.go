package policy

import "testing"

func TestCombatExecutionContractMetadata(t *testing.T) {
	for _, value := range []string{"", LegacyCombatExecution, CheckedGroundCombatExecution} {
		got, err := CombatExecutionContract(value)
		if err != nil || got == "" {
			t.Fatal(value, got, err)
		}
		if value == "" && got != LegacyCombatExecution {
			t.Fatal("historical contract changed")
		}
	}
	for _, value := range []string{"unknown", "legacy_v2", "checked_ground_crouch_v1 "} {
		if _, err := CombatExecutionContract(value); err == nil {
			t.Fatal("unknown contract accepted", value)
		}
	}
}
