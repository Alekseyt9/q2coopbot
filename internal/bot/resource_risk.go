package bot

import (
	"math"

	"q2coopbot/internal/quake"
)

// ResourceRisk considers only monsters in the current server observation.
// Static geometry and observed doors predict exposure, not actual future hits.
type ResourceRisk struct {
	Entity   int        `json:"entity"`
	Class    string     `json:"class"`
	State    string     `json:"state"`
	Reason   string     `json:"reason"`
	Enemy    int        `json:"enemy,omitempty"`
	Position quake.Vec3 `json:"position"`
}

func (p *Planner) resourceDetourAllowed(s quake.Snapshot, item quake.Object, at quake.Vec3, fromMemory bool) bool {
	// Picking up a useful visible item already beside us is not a long detour.
	if len(s.Enemies) == 0 || !fromMemory && quake.Distance(s.Self, at) <= 96 {
		return true
	}
	d := &ResourceRisk{Entity: item.ID, Class: item.Class, State: "deferred", Reason: "route_unavailable", Position: at}
	p.World.ResourceRisk = d
	if p.Nav == nil || !p.World.Geometry.HasCollision() {
		return false
	}
	route, ok := p.Nav.Route(s.Self, at)
	if !ok {
		return false
	}
	exposed := func(point quake.Vec3, eyeHeight float64) bool {
		eye := point
		eye[2] += eyeHeight
		for _, enemy := range s.Enemies {
			origin := enemy.AimPoint()
			if p.World.Geometry.ClearShot(origin, eye) && !p.World.Geometry.DoorShotBlocked(s.Movers, origin, eye) {
				d.Enemy, d.Position = enemy.ID, point
				return true
			}
		}
		return false
	}
	if exposed(s.Self, s.EyePoint()[2]-s.Self[2]) {
		d.Reason = "current_exposure"
		return false
	}
	points := make([]quake.Vec3, 0, len(route)+1)
	for _, wp := range route {
		points = append(points, wp.Position)
	}
	points = append(points, at)
	previous := s.Self
	for _, point := range points {
		steps := max(1, int(math.Ceil(quake.Distance(previous, point)/32)))
		for step := 1; step <= steps; step++ {
			fraction := float64(step) / float64(steps)
			var sample quake.Vec3
			for axis := range sample {
				sample[axis] = previous[axis] + (point[axis]-previous[axis])*fraction
			}
			if exposed(sample, 22) {
				d.Reason = "route_exposure"
				return false
			}
		}
		previous = point
	}
	d.State, d.Reason = "allowed", "no_route_exposure"
	return true
}
