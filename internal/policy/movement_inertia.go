package policy

import (
	"fmt"
	"math"
)

const MovementInertiaVersion = "combat_previous_command_inertia_v1"
const movementInertiaEpsilon = 1.0 / 4096

type MovementInertiaFile struct {
	Version string  `json:"version"`
	Rho     float64 `json:"rho"`
}

func validateMovementInertia(f PPOFile) error {
	if f.MovementInertia == nil {
		return nil
	}
	h := f.MovementInertia
	if h.Version != MovementInertiaVersion || math.IsNaN(h.Rho) || math.IsInf(h.Rho, 0) || h.Rho < 0 || h.Rho > .95 ||
		f.Features != EquippedWeaponFeatureVersion || f.SpatialAim == nil || f.EntityAttention != nil ||
		f.MemoryRecovery != nil || f.DiagnosticMovementProbe != nil || f.DiagnosticContinuation != nil || f.ExecutionMode != "" {
		return fmt.Errorf("invalid movement inertia contract")
	}
	return nil
}

// State-only feedback after the existing motor. Std projection is explicit in
// the policy JSON; no RNG, memory update or hidden noise process is added here.
func (p *PPO) movementInertiaRaw(o Observation, raw []float64) ([]float64, error) {
	h := p.file.MovementInertia
	if h == nil || h.Rho == 0 {
		return raw, nil
	}
	if len(raw) < 2 {
		return nil, fmt.Errorf("invalid inertia raw width")
	}
	previous := [2]float64{float64(o.PreviousCommand.Forward) / 400, float64(o.PreviousCommand.Side) / 400}
	out := append([]float64(nil), raw...)
	for i, v := range previous {
		if math.IsNaN(raw[i]) || math.IsInf(raw[i], 0) {
			return nil, fmt.Errorf("nonfinite inertia mean")
		}
		v = math.Max(-1, math.Min(1, v))
		blended := (1-h.Rho)*math.Tanh(raw[i]) + h.Rho*v
		out[i] = math.Atanh(math.Max(-1+movementInertiaEpsilon, math.Min(1-movementInertiaEpsilon, blended)))
	}
	return out, nil
}
