package quake

import "math"

// GroundPathStatus is a bounded static-world diagnostic, not a movement
// permission. It sweeps the hull continuously and samples flat support every
// two units. Narrow gaps between support samples and dynamic entities are not
// ruled out. Steps/slopes are deliberately outside the flat friction model.
func (m *MapInfo) GroundPathStatus(from, to Vec3, ducked bool) string {
	if !m.MovementComplete() {
		return "unknown_geometry"
	}
	for axis := range from {
		if math.IsNaN(from[axis]) || math.IsInf(from[axis], 0) || math.IsNaN(to[axis]) || math.IsInf(to[axis], 0) {
			return "invalid_path"
		}
	}
	distance := Horizontal(from, to)
	if distance > 64 || math.Abs(to[2]-from[2]) > 0.001 {
		return "unsupported_path"
	}
	clear := false
	if ducked {
		clear = m.CrouchMoveClear(from, to)
	} else {
		clear = m.PlayerMoveClear(from, to)
	}
	if !clear {
		return "static_hull_blocked"
	}
	steps := max(1, int(math.Ceil(distance/2)))
	for i := 0; i <= steps; i++ {
		fraction := float64(i) / float64(steps)
		point := Vec3{from[0] + (to[0]-from[0])*fraction, from[1] + (to[1]-from[1])*fraction, from[2]}
		if surface := m.GroundFrictionStatus(point); surface != "dry_flat" {
			return surface
		}
	}
	return "static_sampled_clear"
}
