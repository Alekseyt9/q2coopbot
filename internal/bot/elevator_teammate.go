package bot

import (
	"math"
	"q2coopbot/internal/quake"
)

// At the upper stop, keep the lift controller active when a visible teammate
// occupies the exit. Do not improvise a sidestep on unverified mover support.
func elevatorTeammateBlocksExit(s quake.Snapshot, target quake.Vec3, model quake.BSPModel, mover quake.Mover) bool {
	return elevatorTeammateBlocksPath(s, target, model, mover, 30)
}
func elevatorTeammateBlocksPath(s quake.Snapshot, target quake.Vec3, model quake.BSPModel, mover quake.Mover, lookahead float64) bool {
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
	for step := 0.0; step <= math.Min(lookahead, distance); step += 2 {
		if math.Abs(s.Self[0]+dx*step-s.Teammate[0]) < 34 && math.Abs(s.Self[1]+dy*step-s.Teammate[1]) < 34 {
			return true
		}
	}
	return false
}
func (p *Planner) elevatorExitMove(cmd quake.UserCmd, s quake.Snapshot, target quake.Vec3, model quake.BSPModel, mover quake.Mover) quake.UserCmd {
	if p.elevator != nil && len(p.elevator.bypass) > 0 {
		if moved, ok := p.elevatorBypass(cmd, s, target, model, mover); ok {
			return moved
		}
	}
	if elevatorTeammateBlocksExit(s, target, model, mover) {
		if p.elevator != nil && p.elevator.waitAnchor != nil {
			return p.elevatorWaitInside(cmd, s, model, mover)
		}
		if moved, ok := p.elevatorBypass(cmd, s, target, model, mover); ok {
			return moved
		}
		return p.elevatorWaitInside(cmd, s, model, mover)
	}
	if p.elevator != nil && p.elevator.waitAnchor != nil {
		// Keep waiting while the teammate still occupies the actual exit,
		// even after retreating beyond the short forward collision probe.
		if elevatorTeammateBlocksPath(s, target, model, mover, 100) {
			return p.elevatorWaitInside(cmd, s, model, mover)
		}
		p.elevator.waitAnchor = nil
	}
	return elevatorMove(cmd, s, target)
}

// Waiting at the outer edge may leave the native platform trigger and allow
// the tall platform to descend onto the player. Retreat onto verified support.
func (p *Planner) elevatorWaitInside(cmd quake.UserCmd, s quake.Snapshot, model quake.BSPModel, mover quake.Mover) quake.UserCmd {
	if p.elevator != nil && p.elevator.waitAnchor == nil {
		anchor := s.Self
		for axis := 0; axis < 2; axis++ {
			anchor[axis] = math.Max(model.Min[axis]+mover.Origin[axis]+32, math.Min(model.Max[axis]+mover.Origin[axis]-32, anchor[axis]))
		}
		if p.elevatorBypassClear(s, mover, s.Self, anchor) {
			p.elevator.waitAnchor = &anchor
		}
	}
	if p.elevator != nil && p.elevator.waitAnchor != nil && quake.Horizontal(s.Self, *p.elevator.waitAnchor) > 4 && p.elevatorBypassClear(s, mover, s.Self, *p.elevator.waitAnchor) {
		p.World.Elevator = "exit_teammate_retreat"
		cmd.Up = 0
		return worldMove(cmd, s, p.elevator.waitAnchor[0]-s.Self[0], p.elevator.waitAnchor[1]-s.Self[1], 60, false)
	}
	p.World.Elevator = "exit_teammate_wait"
	cmd.Forward, cmd.Side, cmd.Up = 0, 0, 0
	return cmd
}
