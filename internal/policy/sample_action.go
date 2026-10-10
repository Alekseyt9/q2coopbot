package policy

import (
	"fmt"
	"math"
)

// ActionForSample verifies the sampled action contract without running a
// network. Likelihood, critic and memory checks belong to the CUDA finalizer.
func (p *PPO) ActionForSample(o Observation, s Sample) (Action, error) {
	if s.Version != p.version || s.Vertical < 0 || s.Vertical > 2 ||
		s.AimMode < 0 || s.AimMode > 1 || p.file.AimModeHead == "" && s.AimMode != 0 ||
		p.file.WeaponHead == "" && s.Weapon != 0 || p.file.TargetHead == "" && s.Target != 0 {
		return Action{}, fmt.Errorf("sample contract differs")
	}
	for _, z := range s.Latent {
		if math.IsNaN(z) || math.IsInf(z, 0) {
			return Action{}, fmt.Errorf("nonfinite latent")
		}
	}
	scale := 180.0
	if s.AimMode == 1 {
		scale = PrecisionFineDegrees
	}
	a := Action{Version: ActionVersion, Identity: o.Identity, Forward: math.Tanh(s.Latent[0]), Side: math.Tanh(s.Latent[1]), YawDelta: scale * math.Tanh(s.Latent[2]), PitchDelta: scale * math.Tanh(s.Latent[3]), Attack: s.Attack, Vertical: []string{"release", "jump", "crouch"}[s.Vertical], AimMode: s.AimMode}
	if s.Target < 0 || s.Target > len(TargetEnemies(o)) {
		return Action{}, fmt.Errorf("unavailable target")
	}
	if s.Target > 0 {
		enemy := TargetEnemies(o)[s.Target-1]
		a.TargetEntity = enemy.ID
		if enemy.Track != nil {
			a.TargetTrack = *enemy.Track
		}
	}
	if p.file.WeaponHead != "" {
		mask, err := WeaponAvailability(o)
		if err != nil || s.Weapon < 0 || s.Weapon >= len(weaponNames) || !mask[s.Weapon] {
			return Action{}, fmt.Errorf("unavailable weapon")
		}
		a.Weapon = weaponNames[s.Weapon]
	}
	_, err := Command(o, a, [3]int16{})
	return a, err
}
