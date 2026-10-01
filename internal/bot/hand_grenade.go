package bot

import (
	"math"
	"q2coopbot/internal/quake"
	"strings"
)

func isHandGrenade(weapon string) bool {
	return strings.EqualFold(weapon, "Grenades") || strings.Contains(strings.ToLower(weapon), "/v_handgr/")
}

// Hand grenades fire on release, unlike ordinary guns. Until a trajectory and
// splash-safe throw is available, never start one through the generic attack
// policy. Releasing also prevents an already armed grenade being held forever.
func (p *Planner) guardHandGrenade(s quake.Snapshot, cmd quake.UserCmd) quake.UserCmd {
	p.World.GrenadePrediction = nil
	if s.Health > 0 && isHandGrenade(s.Weapon) {
		p.World.GrenadePrediction = p.predictGrenadeCandidate(s)
		cmd.Buttons &^= 1
		p.World.Command.LimitReason = "hand_grenade_requires_safe_throw"
		if s.GunFrame >= 1 && s.GunFrame <= 15 {
			// Finish the native firing cycle even if its target disappeared.
			// Navigation must not swivel an already primed throw toward a player.
			oldYaw, forward, side := cmd.Yaw, float64(cmd.Forward), float64(cmd.Side)
			cmd.Pitch = s.ViewAngles[0] - s.DeltaAngles[0]
			cmd.Yaw = s.ViewAngles[1] - s.DeltaAngles[1]
			cmd.Roll = s.ViewAngles[2] - s.DeltaAngles[2]
			// Preserve the world-space movement already checked against BSP:
			// its local forward/right basis changes when the aim is restored.
			angle := (float64(oldYaw) - float64(cmd.Yaw)) * 2 * math.Pi / 65536
			cmd.Forward = int16(math.Round(forward*math.Cos(angle) + side*math.Sin(angle)))
			cmd.Side = int16(math.Round(-forward*math.Sin(angle) + side*math.Cos(angle)))
			p.World.Command.AimSource = "grenade_release_pose"
			p.World.Command.LimitReason = "hand_grenade_finish_preparation"
			if s.GunFrame == 11 {
				p.World.Command.LimitReason = "hand_grenade_release"
			}
			if s.GunFrame >= 12 {
				p.World.Command.LimitReason = "hand_grenade_wait_recovery"
			}
		}
	}
	return cmd
}
