package bot

import (
	"math"
	"q2coopbot/internal/quake"
)

// Certify only the model's own static wall-contact command. No target,
// acceleration or route is supplied here; the native engine resolves contact.
func (p *Planner) checkedCombatWallMotion(s quake.Snapshot, cmd quake.UserCmd) bool {
	g := p.World.Geometry
	if cmd.Up > 0 || !g.MovementComplete() {
		return false
	}
	prediction := predictPmoveGround(s, cmd, g)
	if prediction == nil || len(prediction.Contacts) == 0 {
		return false
	}
	return p.checkedCombatMotionPaths(s, cmd, prediction)
}

func (p *Planner) checkedCombatMotionPaths(s quake.Snapshot, cmd quake.UserCmd, prediction *PmoveGroundPrediction) bool {
	g := p.World.Geometry
	seen := map[*PmoveGroundPrediction]bool{}
	for path := prediction; path != nil; path = path.Alternative {
		if seen[path] || len(seen) >= 2 || path.SupportAmbiguous && path.Alternative == nil {
			return false
		}
		seen[path] = true
		if len(path.Segments) == 0 {
			return false
		}
		current := s.Self
		for _, segment := range path.Segments {
			if segment[0] != current {
				return false
			}
			current = segment[1]
			if g.MotionPathHazard(segment[0], segment[1], cmd.Up < 0) != "" {
				return false
			}
			for _, mover := range s.Movers {
				if g.MoverNearMotion(mover, segment[0], segment[1]) {
					return false
				}
			}
			for _, body := range s.CollisionBodies {
				if quake.BodyNearMotion(body, segment[0], segment[1]) {
					return false
				}
			}
			dx, dy := segment[1][0]-segment[0][0], segment[1][1]-segment[0][1]
			if _, hazard := g.DoorMoveBlockStep(s.Movers, segment[0], dx, dy, math.Hypot(dx, dy)); hazard != "" {
				return false
			}
		}
		if current != path.Position {
			return false
		}
	}
	return len(seen) > 0
}
