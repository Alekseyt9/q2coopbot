package bot

import (
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
	if s.Health > 0 && isHandGrenade(s.Weapon) {
		cmd.Buttons &^= 1
		p.World.Command.LimitReason = "hand_grenade_requires_safe_throw"
	}
	return cmd
}
