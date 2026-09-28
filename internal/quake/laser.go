package quake

import "math"

type laserBeam struct {
	start, end Vec3
	damage     int
}

// Lasers with START_ON are treated as hazards until live state can prove they
// are off. A switchable beam may therefore be conservatively blocked.
func (m *MapInfo) initStaticLasers() {
	if m == nil || !m.HasCollision() {
		return
	}
	for _, e := range m.Entities {
		if e.Class != "target_laser" || e.SpawnFlags&1 == 0 || e.Target != "" || e.Damage < 50 {
			continue
		}
		yaw, pitch := e.Angles[1]*math.Pi/180, e.Angles[0]*math.Pi/180
		direction := Vec3{math.Cos(pitch) * math.Cos(yaw), math.Cos(pitch) * math.Sin(yaw), -math.Sin(pitch)}
		full := Vec3{e.Origin[0] + 2048*direction[0], e.Origin[1] + 2048*direction[1], e.Origin[2] + 2048*direction[2]}
		length := 2048.0
		if !m.ClearShot(e.Origin, full) {
			lo, hi := 0.0, length
			for i := 0; i < 14; i++ {
				mid := (lo + hi) / 2
				point := Vec3{e.Origin[0] + mid*direction[0], e.Origin[1] + mid*direction[1], e.Origin[2] + mid*direction[2]}
				if m.ClearShot(e.Origin, point) {
					lo = mid
				} else {
					hi = mid
				}
			}
			length = lo
		}
		if length < 8 {
			continue
		}
		m.laserBeams = append(m.laserBeams, laserBeam{start: e.Origin, end: Vec3{e.Origin[0] + length*direction[0], e.Origin[1] + length*direction[1], e.Origin[2] + length*direction[2]}, damage: e.Damage})
	}
}

func (m *MapInfo) HasStaticLethalLasers() bool { return m != nil && len(m.laserBeams) > 0 }

// LaserMoveHazard checks the standing player's full hull plus a small margin
// along a planned ground segment. It is conservative near a beam end.
func (m *MapInfo) LaserMoveHazard(from, to Vec3) bool {
	if m == nil {
		return false
	}
	distance := Horizontal(from, to)
	steps := int(math.Ceil(distance / 4))
	if steps < 1 {
		steps = 1
	}
	for _, beam := range m.laserBeams {
		for i := 0; i <= steps; i++ {
			t := float64(i) / float64(steps)
			p := Vec3{from[0] + t*(to[0]-from[0]), from[1] + t*(to[1]-from[1]), from[2] + t*(to[2]-from[2])}
			dx, dy := beam.end[0]-beam.start[0], beam.end[1]-beam.start[1]
			den := dx*dx + dy*dy
			if den < 1 {
				continue
			}
			u := math.Max(0, math.Min(1, ((p[0]-beam.start[0])*dx+(p[1]-beam.start[1])*dy)/den))
			x, y, z := beam.start[0]+u*dx, beam.start[1]+u*dy, beam.start[2]+u*(beam.end[2]-beam.start[2])
			if math.Abs(p[0]-x) <= 22 && math.Abs(p[1]-y) <= 22 && z >= p[2]-28 && z <= p[2]+36 {
				return true
			}
		}
	}
	return false
}
