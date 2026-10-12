package quake

// StepSlidePlayer compares the native ordinary slide with an 18-unit raised
// slide and landing. It predicts the collision phase only; it does not apply
// friction, acceleration, gravity, PM timers or network snapping. Dynamic
// entities and lethal surfaces must be checked by the caller over all segments.
func (m *MapInfo) StepSlidePlayer(from, velocity Vec3, seconds float64, ducked bool) (PlayerSlide, bool) {
	return m.stepSlidePlayer(from, velocity, seconds, ducked, m.TracePlayer)
}

func (m *MapInfo) StepSlidePlayerWithMovers(from, velocity Vec3, seconds float64, ducked bool, movers []Mover) (PlayerSlide, bool) {
	return m.stepSlidePlayer(from, velocity, seconds, ducked, func(a, b Vec3, duck bool) PlayerTrace {
		return m.TracePlayerWithMovers(a, b, duck, movers)
	})
}

func (m *MapInfo) stepSlidePlayer(from, velocity Vec3, seconds float64, ducked bool, trace func(Vec3, Vec3, bool) PlayerTrace) (PlayerSlide, bool) {
	down, ok := m.slidePlayer(from, velocity, seconds, ducked, trace)
	if !ok {
		return down, false
	}
	raised := from
	raised[2] += 18
	position := trace(raised, raised, ducked)
	if !position.Valid {
		return down, false
	}
	if position.AllSolid || position.StartSolid {
		return down, true
	}
	up, ok := m.slidePlayer(raised, velocity, seconds, ducked, trace)
	if !ok {
		return down, false
	}
	landing := up.Position
	landing[2] -= 18
	tr := trace(up.Position, landing, ducked)
	if !tr.Valid || tr.StartSolid || tr.AllSolid {
		// Native allsolid step candidates cannot certify a safe trajectory.
		return down, false
	}
	up.Segments = append([][2]Vec3{{from, raised}}, up.Segments...)
	up.Segments = append(up.Segments, [2]Vec3{up.Position, tr.End})
	up.Position = tr.End
	horizontalSquared := func(to Vec3) float64 {
		x, y := to[0]-from[0], to[1]-from[1]
		return x*x + y*y
	}
	if horizontalSquared(down.Position) > horizontalSquared(up.Position) || tr.Normal[2] < .7 {
		return down, true
	}
	up.Velocity[2] = down.Velocity[2]
	return up, true
}
