package bot

import (
	"q2coopbot/internal/quake"
	"strings"
)

type machinegunBurst struct {
	mapName                                  string
	start, lastAttack, lastFrame, pauseUntil int
}

func isMachinegun(weapon string) bool {
	return strings.EqualFold(weapon, "Machinegun") || strings.Contains(strings.ToLower(weapon), "/v_machn/")
}

// Baseq2 accumulates 1.5 degrees of pitch kick for each held machinegun shot.
// Limit attack to three server frames, then release for two server frames.
// All command paths pass here, so movement and target switches cannot bypass it.
func (p *Planner) limitMachinegunBurst(s quake.Snapshot, cmd quake.UserCmd) quake.UserCmd {
	b := &p.machinegunBurst
	if !isMachinegun(s.Weapon) || s.Health <= 0 || s.Frame <= 0 {
		*b = machinegunBurst{}
		return cmd
	}
	if b.mapName != s.Map || s.Frame < b.lastFrame {
		*b = machinegunBurst{mapName: s.Map}
	}
	b.lastFrame = s.Frame
	if cmd.Buttons&1 == 0 {
		if s.Frame-b.lastAttack >= 2 {
			b.start = 0
		}
		return cmd
	}
	if b.start != 0 && s.Frame-b.start >= 3 {
		b.start = 0
		b.pauseUntil = s.Frame + 2
	}
	if s.Frame < b.pauseUntil {
		cmd.Buttons &^= 1
		p.World.Command.LimitReason = "machinegun_burst_pause"
		return cmd
	}
	if b.start == 0 {
		b.start = s.Frame
	}
	b.lastAttack = s.Frame
	return cmd
}
