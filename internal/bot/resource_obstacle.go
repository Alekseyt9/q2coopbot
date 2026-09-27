package bot

import (
	"math"
	"q2coopbot/internal/quake"
)

// A bounded ground step around an observed blocker; never extends the health
// deadline and never substitutes a jump for missing floor or a closed door.
func (p *Planner) resourceObstacleStep(s quake.Snapshot, dx, dy float64) (float64, float64, bool) {
	length := math.Hypot(dx, dy)
	if p.World.Goal != "recover_health" || !s.OnGround || s.Frame-p.healthAt < 5 || length < 1 || p.Nav == nil || !p.World.Geometry.HasCollision() {
		return 0, 0, false
	}
	ux, uy := dx/length, dy/length
	for _, wp := range p.World.Route {
		if wp.Kind != 2 || math.Abs(wp.Position[2]-s.Self[2]) > 16 {
			return 0, 0, false
		}
	}
	var blocker *quake.Object
	for i := range s.Obstacles {
		e := &s.Obstacles[i]
		ex, ey := e.Origin[0]-s.Self[0], e.Origin[1]-s.Self[1]
		if math.Abs(e.Origin[2]-s.Self[2]) <= 32 && math.Hypot(ex, ey) < 64 {
			blocker = e
			break
		}
	}
	if blocker == nil {
		return 0, 0, false
	}
	for _, turn := range []float64{math.Pi / 12, -math.Pi / 12, math.Pi / 6, -math.Pi / 6, math.Pi / 3, -math.Pi / 3, math.Pi / 2, -math.Pi / 2} {
		x, y := 16*(ux*math.Cos(turn)-uy*math.Sin(turn)), 16*(ux*math.Sin(turn)+uy*math.Cos(turn))
		end := s.Self
		end[0] += x
		end[1] += y
		if p.World.Geometry.GroundMoveHazardStep(p.Nav, s.Self, x, y, 16) != "" || p.World.Geometry.DoorMoveHazard(s.Movers, s.Self, x, y) != "" {
			continue
		}
		clear := true
		for _, e := range s.Obstacles {
			// Soldier half-width16 plus player half-width16. Other monsters
			// keep the conservative circular margin until hulls are decoded.
			for _, fraction := range []float64{0.5, 1} {
				at := s.Self
				at[0] += x * fraction
				at[1] += y * fraction
				blocked := quake.Horizontal(at, e.Origin) < 40
				if e.Class == "monster_soldier" {
					blocked = math.Abs(at[0]-e.Origin[0]) < 33 && math.Abs(at[1]-e.Origin[1]) < 33
				}
				if math.Abs(e.Origin[2]-at[2]) <= 40 && blocked {
					clear = false
					break
				}
			}
		}
		if s.Teammate != nil && math.Abs((*s.Teammate)[2]-end[2]) <= 40 && quake.Horizontal(end, *s.Teammate) < 40 {
			clear = false
		}
		if clear {
			return x, y, true
		}
	}
	return 0, 0, false
}
