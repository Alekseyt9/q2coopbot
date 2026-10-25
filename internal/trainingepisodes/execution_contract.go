package trainingepisodes

import (
	"encoding/json"
	"fmt"
	"os"
	"q2coopbot/internal/policy"
)

// Metadata only: no neural forward pass is needed to admit a frozen schedule.
func ValidatePlanExecutionContract(p *Plan) error {
	contract, err := policy.CombatExecutionContract(p.ExecutionContract)
	if err != nil {
		return err
	}
	if p.ModelPath != "" {
		data, err := os.ReadFile(p.ModelPath)
		if err != nil {
			return err
		}
		var header struct {
			Contract string `json:"combat_execution_contract"`
		}
		if err = json.Unmarshal(data, &header); err != nil {
			return err
		}
		modelContract, err := policy.CombatExecutionContract(header.Contract)
		if err != nil {
			return err
		}
		if contract != modelContract {
			return fmt.Errorf("model execution contract differs from plan")
		}
	}
	if contract != policy.LegacyCombatExecution {
		if p.ModelPath == "" || !p.CheckedWallSlide || p.DiagnosticStoppedCrouch || p.DiagnosticCheckedGroundRequest || p.DiagnosticNominalAim {
			return fmt.Errorf("checked execution contract requires exclusive learned plan with checked wall slide")
		}
		for _, task := range p.Tasks {
			if len(task.Modes) != 1 || task.Modes[0] != "learned" {
				return fmt.Errorf("checked execution contract requires learned-only tasks")
			}
		}
	}
	return nil
}
