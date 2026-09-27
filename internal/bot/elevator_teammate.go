package bot

import (
	"math"
	"q2coopbot/internal/quake"
)

// At the upper stop, keep the lift controller active when a visible teammate
// occupies the exit. Do not improvise a sidestep on unverified mover support.
func elevatorTeammateBlocksExit(s quake.Snapshot, target quake.Vec3, model quake.BSPModel, mover quake.Mover) bool {
	if s.Teammate == nil || !s.OnGround || math.Abs(mover.Origin[2]-model.Origin[2]) > 0.125 || math.Abs(s.Self[2]-s.Teammate[2]) > 48 {
		return false
	}
	dx, dy := target[0]-s.Self[0], target[1]-s.Self[1]
	distance := math.Hypot(dx, dy)
	if distance < 1 {
		return false
	}
	dx, dy = dx/distance, dy/distance
	if (s.Teammate[0]-s.Self[0])*dx+(s.Teammate[1]-s.Self[1])*dy <= 0 {
		return false
	}
	for step := 0.0; step <= math.Min(30, distance); step += 2 {
		if math.Abs(s.Self[0]+dx*step-s.Teammate[0]) < 34 && math.Abs(s.Self[1]+dy*step-s.Teammate[1]) < 34 {
			return true
		}
	}
	return false
}
func (p *Planner) elevatorExitMove(cmd quake.UserCmd, s quake.Snapshot, target quake.Vec3, model quake.BSPModel, mover quake.Mover) quake.UserCmd {
	if elevatorTeammateBlocksExit(s, target, model, mover) {
		p.World.Elevator = "exit_teammate_wait"
		cmd.Forward, cmd.Side, cmd.Up = 0, 0, 0
		return cmd
	}
	return elevatorMove(cmd, s, target)
}
