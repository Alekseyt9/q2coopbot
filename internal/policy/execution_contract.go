package policy

import "fmt"

const LegacyCombatExecution = "legacy_v1"
const CheckedGroundCombatExecution = "checked_ground_crouch_v1"
const StockWeaponsFireCombatExecution = "stock_weapons_fire_v1"

// Empty historical metadata denotes the original environment, never a new one.
func CombatExecutionContract(value string) (string, error) {
	if value == "" {
		return LegacyCombatExecution, nil
	}
	if value != LegacyCombatExecution && value != CheckedGroundCombatExecution && value != StockWeaponsFireCombatExecution {
		return "", fmt.Errorf("unknown combat execution contract %q", value)
	}
	return value, nil
}

func (p *PPO) CombatExecutionContract() string {
	contract, _ := CombatExecutionContract(p.file.CombatExecution)
	return contract
}
