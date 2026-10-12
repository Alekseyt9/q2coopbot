package quake

import "math"

// MotionPathHazard checks a bounded hull sweep and sampled static support.
// It is not a proof against sub-sample gaps or moving/unobserved entities.
func (m *MapInfo) MotionPathHazard(from, to Vec3, ducked bool) string {
	if !m.MovementComplete() {
		return "unknown_geometry"
	}
	for axis := range from {
		if math.IsNaN(from[axis]) || math.IsNaN(to[axis]) || math.IsInf(from[axis], 0) || math.IsInf(to[axis], 0) {
			return "invalid_path"
		}
	}
	distance := Distance(from, to)
	if distance > 64 || math.Abs(to[2]-from[2]) > 18.5 {
		return "unsupported_path"
	}
	clear := m.PlayerMoveClear(from, to)
	if ducked {
		clear = m.CrouchMoveClear(from, to)
	}
	if !clear {
		return "static_hull_blocked"
	}
	if m.LaserMoveHazard(from, to) {
		return "static_laser_hazard"
	}
	steps := max(1, int(math.Ceil(distance)))
	for i := 0; i <= steps; i++ {
		point := from
		for axis := range point {
			point[axis] += (to[axis] - from[axis]) * float64(i) / float64(steps)
		}
		if m.PlayerTouchesHazard(point) {
			return "lethal_contents"
		}
		drop, normal, _, ok := m.collision.groundContactInset(point, 18.5, 0)
		if !ok || normal[2] < .9999 {
			return "no_flat_support"
		}
		floor := point
		floor[2] -= drop
		floor[2] += .125
		if m.GroundFrictionStatus(floor) != "dry_flat" {
			return "unsupported_surface"
		}
	}
	return ""
}

// BodyNearMotion includes a one-eighth-unit position uncertainty around each
// observed solid box. True requires separate treatment of body-contact paths.
func BodyNearMotion(body Object, from, to Vec3) bool {
	lo, hi, ok := body.SolidBounds()
	if !ok {
		return body.Solid != 0
	}
	for axis := range lo {
		lo[axis] += body.Origin[axis] - .125
		hi[axis] += body.Origin[axis] + .125
	}
	return sweptBoxAABB(from, to, lo, hi, Vec3{-16, -16, -24}, Vec3{16, 16, 32})
}
