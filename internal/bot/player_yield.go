package bot

import (
	"math"
	"q2coopbot/internal/quake"
)

const followStandOff = 128.0
const playerPersonalSpace = 96.0

// A close player with no room to pass must not be pinned by cover_teammate.
// Retreat to restore personal space, using a side pocket if needed.
// Re-evaluate geometry and the observed player on every command.
func (p *Planner) playerYieldCommand(s quake.Snapshot, cmd quake.UserCmd) (quake.UserCmd, bool) {
	g := p.World.Geometry
	if s.Teammate == nil || s.Health <= 0 || !s.OnGround || p.testSetupHold || p.elevator != nil || p.jump != nil ||
		(p.World.Goal != "cover_teammate" && p.World.Goal != "follow_teammate") || !g.HasCollision() || math.Abs(s.Self[2]-s.Teammate[2]) > 24 {
		return cmd, false
	}
	dx, dy := s.Self[0]-s.Teammate[0], s.Self[1]-s.Teammate[1]
	d := math.Hypot(dx, dy)
	if d < 1 || d > 112 {
		return cmd, false
	}
	ux, uy := dx/d, dy/d
	m := p.shotTeammateMotion
	approaching := m.known && m.frame == s.Frame && m.entity == s.TeammateEntity && m.velocity[0]*ux+m.velocity[1]*uy > 20
	if !approaching && d >= playerPersonalSpace {
		return cmd, false
	}
	// Side pockets let the player pass. When neither is clear, move farther
	// along the corridor instead of insisting on a lateral bypass.
	directions := [][2]float64{{ux, uy}, {-uy, ux}, {uy, -ux}, {0, math.Copysign(1, uy)}, {math.Copysign(1, ux), 0}}
	for _, dir := range directions {
		at := s.Self
		clear := true
		for step := 0; step < 7; step++ {
			next := quake.Vec3{at[0] + dir[0]*8, at[1] + dir[1]*8, at[2]}
			if !g.PlayerMoveClear(at, next) {
				clear = false
				break
			}
			if _, ok := g.GroundDrop(next, 18); !ok {
				clear = false
				break
			}
			if _, reason := g.DoorMoveBlockStep(s.Movers, at, dir[0], dir[1], 8); reason != "" {
				clear = false
				break
			}
			if quake.Horizontal(next, *s.Teammate) < d-0.5 {
				clear = false
				break
			}
			for _, obstacle := range s.Obstacles {
				if obstacle.ID != s.TeammateEntity && math.Abs(obstacle.Origin[2]-next[2]) < 48 && math.Abs(obstacle.Origin[0]-next[0]) < 36 && math.Abs(obstacle.Origin[1]-next[1]) < 36 {
					clear = false
					break
				}
			}
			if !clear {
				break
			}
			at = next
		}
		if !clear {
			continue
		}
		cmd.Up = 0
		p.World.Command.MoveSource = "teammate_yield"
		p.World.Command.Skill = "teammate_yield"
		p.World.Command.MoveLimitReason = "verified_player_yield"
		return worldMove(cmd, s, dir[0], dir[1], 80, p.World.Command.AimSource == "enemy"), true
	}
	return cmd, false
}
