package bot

import (
	"math"

	"q2coopbot/internal/policy"
	"q2coopbot/internal/quake"
)

// diagnosticNominalCombatAim is an opt-in experiment, never a learned action.
// It keeps the selected observed target and all discrete decisions. It does not
// predict projectile lead, future recoil, hidden positions or ballistic spread.
func diagnosticNominalCombatAim(o policy.Observation, a policy.Action, cmd quake.UserCmd, delta [3]int16) (quake.UserCmd, []policy.Intervention) {
	if !directHitscanWeapon(o.Weapon) {
		return cmd, nil
	}
	for _, e := range o.Enemies {
		if a.TargetEntity <= 0 || e.ID != a.TargetEntity || e.Track == nil && a.TargetTrack != 0 || e.Track != nil && *e.Track != a.TargetTrack {
			continue
		}
		direction, known, err := policy.ObservedAimDirection(o, e)
		if err != nil || !known {
			return cmd, nil
		}
		yaw := math.Atan2(direction[1], direction[0]) * 180 / math.Pi
		pitch := -math.Atan2(direction[2], math.Hypot(direction[0], direction[1])) * 180 / math.Pi
		if o.KickAngles != nil {
			pitch -= (*o.KickAngles)[0]
			yaw -= (*o.KickAngles)[1]
		}
		if math.IsNaN(yaw) || math.IsInf(yaw, 0) || math.IsNaN(pitch) || math.IsInf(pitch, 0) {
			return cmd, nil
		}
		pitch = math.Max(-89, math.Min(89, pitch))
		short := func(angle float64) int16 { return int16(int64(math.Round(angle * 65536 / 360))) }
		original := cmd
		cmd.Yaw = int16(uint16(short(yaw)) - uint16(delta[1]))
		cmd.Pitch = int16(uint16(short(pitch)) - uint16(delta[0]))
		// Preserve world movement direction when changing its view basis. The
		// command box can require uniform scaling; the following guard remains.
		rotation := float64(int16(uint16(original.Yaw)-uint16(cmd.Yaw))) * 2 * math.Pi / 65536
		f, s := float64(original.Forward), float64(original.Side)
		forward, side := f*math.Cos(rotation)+s*math.Sin(rotation), -f*math.Sin(rotation)+s*math.Cos(rotation)
		scale := math.Max(1, math.Max(math.Abs(forward), math.Abs(side))/400)
		cmd.Forward, cmd.Side = int16(math.Round(forward/scale)), int16(math.Round(side/scale))
		changes := []policy.Intervention{{Component: "yaw", Reason: "diagnostic_nominal_observed_aim"}, {Component: "pitch", Reason: "diagnostic_nominal_observed_aim"}}
		if cmd.Forward != original.Forward || cmd.Side != original.Side {
			changes = append(changes, policy.Intervention{Component: "movement", Reason: "diagnostic_aim_view_basis"})
		}
		return cmd, changes
	}
	return cmd, nil
}
