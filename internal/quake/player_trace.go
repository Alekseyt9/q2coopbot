package quake

import "math"

// PlayerTrace is a swept native pmove hull against static world brushes only.
// It does not include doors, monsters or other dynamic collision entities.
type PlayerTrace struct {
	Fraction             float64
	End, Normal          Vec3
	StartSolid, AllSolid bool
	Valid                bool
}

func (m *MapInfo) TracePlayer(from, to Vec3, ducked bool) PlayerTrace {
	if !m.HasCollision() {
		return PlayerTrace{}
	}
	maxs := Vec3{16, 16, 32}
	if ducked {
		maxs[2] = 4
	}
	return m.collision.tracePlayerHull(from, to, Vec3{-16, -16, -24}, maxs)
}

// TracePlayerWithMovers includes observed brush models at their current pose.
// It does not extrapolate mover motion, include actors, or clear unseen doors.
func (m *MapInfo) TracePlayerWithMovers(from, to Vec3, ducked bool, movers []Mover) PlayerTrace {
	result := m.TracePlayer(from, to, ducked)
	if !result.Valid {
		return result
	}
	maxs := Vec3{16, 16, 32}
	if ducked {
		maxs[2] = 4
	}
	for _, mover := range movers {
		if mover.SolidKnown && mover.Solid == 0 {
			continue
		}
		local, ok := m.moverCollision(mover)
		if !ok {
			return PlayerTrace{}
		}
		a, b := from, to
		for axis, value := range mover.Origin {
			if math.IsNaN(value) || math.IsInf(value, 0) {
				return PlayerTrace{}
			}
			a[axis] -= value
			b[axis] -= value
		}
		// Native CM_TransformedBoxTrace rotates the segment but retains hull extents.
		if mover.Angles != (Vec3{}) {
			local = *m.collision
			local.worldBrushes = local.modelBrushes[mover.Model]
			a = inverseBrushVector(a, mover.Angles)
			b = inverseBrushVector(b, mover.Angles)
		}
		tr := local.tracePlayerHull(a, b, Vec3{-16, -16, -24}, maxs)
		if !tr.Valid {
			return tr
		}
		result.StartSolid = result.StartSolid || tr.StartSolid
		result.AllSolid = result.AllSolid || tr.AllSolid
		if tr.Fraction < result.Fraction {
			result.Fraction = tr.Fraction
			result.Normal = rotateBrushVector(tr.Normal, mover.Angles)
		}
	}
	for axis := range from {
		result.End[axis] = from[axis] + (to[axis]-from[axis])*result.Fraction
	}
	return result
}

// CM_ClipBoxToBrush plane expansion and DIST_EPSILON, including native
// startsolid semantics: exiting a brush does not set allsolid or stop the trace.
// The older boxClear helper is deliberately conservative and remains unchanged.
func (c *CollisionMap) tracePlayerHull(from, to, mins, maxs Vec3) PlayerTrace {
	for axis := range from {
		for _, value := range []float64{from[axis], to[axis], mins[axis], maxs[axis]} {
			if math.IsNaN(value) || math.IsInf(value, 0) {
				return PlayerTrace{}
			}
		}
		if mins[axis] > maxs[axis] {
			return PlayerTrace{}
		}
	}
	tr := PlayerTrace{Fraction: 1, End: to, Valid: true}
	const playerSolid = 1 | 2 | 0x10000 | 0x2000000
	for _, index := range c.worldBrushes {
		brush := c.brushes[index]
		if brush.contents&playerSolid == 0 || brush.count == 0 {
			continue
		}
		enter, leave := -1.0, 1.0
		startOutside, endOutside, rejected := false, false, false
		normal := Vec3{}
		for side := brush.first; side < brush.first+brush.count; side++ {
			plane := c.planes[c.sides[side]]
			d1, d2 := -plane.dist, -plane.dist
			for axis := range from {
				offset := mins[axis]
				if plane.normal[axis] < 0 {
					offset = maxs[axis]
				}
				d1 += (from[axis] + offset) * plane.normal[axis]
				d2 += (to[axis] + offset) * plane.normal[axis]
			}
			startOutside = startOutside || d1 > 0
			endOutside = endOutside || d2 > 0
			if d1 > 0 && d2 >= d1 {
				rejected = true
				break
			}
			if d1 <= 0 && d2 <= 0 {
				continue
			}
			if d1 > d2 {
				fraction := (d1 - .03125) / (d1 - d2)
				if fraction > enter {
					enter, normal = fraction, plane.normal
				}
			} else {
				leave = math.Min(leave, (d1+.03125)/(d1-d2))
			}
		}
		if rejected {
			continue
		}
		if !startOutside {
			tr.StartSolid = true
			tr.AllSolid = tr.AllSolid || !endOutside
			// Native position tests explicitly set fraction=0 for an occupied hull.
			if from == to {
				tr.Fraction = 0
			}
			continue
		}
		if enter < leave && enter > -1 && enter < tr.Fraction {
			tr.Fraction, tr.Normal = math.Max(0, enter), normal
		}
	}
	for axis := range from {
		tr.End[axis] = from[axis] + (to[axis]-from[axis])*tr.Fraction
	}
	return tr
}
