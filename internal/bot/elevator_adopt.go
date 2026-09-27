package bot

import (
	"math"
	"q2coopbot/internal/quake"
)

func (p *Planner) elevatorAtUpperLanding(s quake.Snapshot, model quake.BSPModel, mover quake.Mover) bool {
	if !s.OnGround || math.Abs(mover.Origin[2]-model.Origin[2]) > .125 || math.Abs(s.Self[2]-p.goalPoint[2]) > 16 || elevatorHullClear(s.Self, model, mover.Origin) || p.World.Geometry == nil {
		return false
	}
	// Actual brush support at the player's feet distinguishes the deck from
	// floors below or above the model's often very tall bounding box.
	for _, offset := range []quake.Vec3{{}, {-16, -16, 0}, {-16, 16, 0}, {16, -16, 0}, {16, 16, 0}} {
		at := s.Self
		at[0] += offset[0]
		at[1] += offset[1]
		if drop, ok := p.World.Geometry.MoverFooting(mover, at, .5); ok && drop <= .5 {
			return true
		}
	}
	return false
}
