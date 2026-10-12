package quake

// SolidBounds decodes protocol-34 prediction bounds, as CL_ClipMoveToEntities.
// Zero is nonsolid; 31 denotes a BSP model with separate brush geometry.
func (o Object) SolidBounds() (Vec3, Vec3, bool) {
	if o.Solid == 0 || o.Solid == 31 {
		return Vec3{}, Vec3{}, false
	}
	x := float64(8 * (o.Solid & 31))
	down := float64(8 * ((o.Solid >> 5) & 31))
	up := float64(8*((o.Solid>>10)&63)) - 32
	if x == 0 || down == 0 || up < -down {
		return Vec3{}, Vec3{}, false
	}
	return Vec3{-x, -x, -down}, Vec3{x, x, up}, true
}

// TracePlayerWithEntities includes currently observed protocol collision bodies.
// Pose extrapolation and server-only entities remain outside this snapshot trace.
func (m *MapInfo) TracePlayerWithEntities(from, to Vec3, ducked bool, movers []Mover, bodies []Object) PlayerTrace {
	result := m.TracePlayerWithMovers(from, to, ducked, movers)
	if !result.Valid {
		return result
	}
	maxs := Vec3{16, 16, 32}
	if ducked {
		maxs[2] = 4
	}
	for _, body := range bodies {
		if body.Solid == 0 || body.Solid == 31 {
			continue
		}
		lo, hi, ok := body.SolidBounds()
		if !ok {
			return PlayerTrace{}
		}
		// Use the same convex brush clipper, not a point ray or corner tests.
		planes := []bspPlane{{Vec3{1, 0, 0}, hi[0]}, {Vec3{-1, 0, 0}, -lo[0]},
			{Vec3{0, 1, 0}, hi[1]}, {Vec3{0, -1, 0}, -lo[1]}, {Vec3{0, 0, 1}, hi[2]}, {Vec3{0, 0, -1}, -lo[2]}}
		box := CollisionMap{planes: planes, sides: []uint16{0, 1, 2, 3, 4, 5}, brushes: []bspBrush{{0, 6, 1}}, worldBrushes: []int{0}}
		a, b := from, to
		for axis := range a {
			a[axis] -= body.Origin[axis]
			b[axis] -= body.Origin[axis]
		}
		tr := box.tracePlayerHull(a, b, Vec3{-16, -16, -24}, maxs)
		if !tr.Valid {
			return tr
		}
		result.StartSolid = result.StartSolid || tr.StartSolid
		result.AllSolid = result.AllSolid || tr.AllSolid
		if tr.Fraction < result.Fraction {
			result.Fraction = tr.Fraction
			result.Normal = tr.Normal
		}
	}
	for axis := range from {
		result.End[axis] = from[axis] + (to[axis]-from[axis])*result.Fraction
	}
	return result
}

func (m *MapInfo) StepSlidePlayerWithEntities(from, velocity Vec3, seconds float64, ducked bool, movers []Mover, bodies []Object) (PlayerSlide, bool) {
	return m.stepSlidePlayer(from, velocity, seconds, ducked, func(a, b Vec3, duck bool) PlayerTrace { return m.TracePlayerWithEntities(a, b, duck, movers, bodies) })
}
