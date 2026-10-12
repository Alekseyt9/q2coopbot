package quake

import "math"

// PlayerSlide predicts only the collision phase of a command. Acceleration,
// friction, gravity, 18-unit stepping, pm_time and network snapping belong to
// the caller/native movement model. Segments allow callers to check every
// travelled path against floor, dynamic movers and lethal hazards.
type PlayerSlide struct {
	Position, Velocity Vec3
	Segments           [][2]Vec3
	Contacts           []Vec3
}

func (m *MapInfo) SlidePlayer(from, velocity Vec3, seconds float64, ducked bool) (PlayerSlide, bool) {
	return m.slidePlayer(from, velocity, seconds, ducked, m.TracePlayer)
}

func (m *MapInfo) SlidePlayerWithMovers(from, velocity Vec3, seconds float64, ducked bool, movers []Mover) (PlayerSlide, bool) {
	return m.slidePlayer(from, velocity, seconds, ducked, func(a, b Vec3, duck bool) PlayerTrace {
		return m.TracePlayerWithMovers(a, b, duck, movers)
	})
}

func (m *MapInfo) slidePlayer(from, velocity Vec3, seconds float64, ducked bool, trace func(Vec3, Vec3, bool) PlayerTrace) (PlayerSlide, bool) {
	result := PlayerSlide{Position: from, Velocity: velocity}
	if !m.HasCollision() || math.IsNaN(seconds) || math.IsInf(seconds, 0) || seconds <= 0 || seconds > .1 {
		return result, false
	}
	for _, vector := range []Vec3{from, velocity} {
		for _, value := range vector {
			if math.IsNaN(value) || math.IsInf(value, 0) {
				return result, false
			}
		}
	}
	dot := func(a, b Vec3) float64 { return a[0]*b[0] + a[1]*b[1] + a[2]*b[2] }
	clip := func(v, normal Vec3) Vec3 {
		backoff := dot(v, normal) * 1.01
		for axis := range v {
			v[axis] -= normal[axis] * backoff
			if math.Abs(v[axis]) < .1 {
				v[axis] = 0
			}
		}
		return v
	}
	planes := []Vec3{}
	remaining := seconds
	for bump := 0; bump < 4; bump++ {
		end := result.Position
		for axis := range end {
			end[axis] += result.Velocity[axis] * remaining
		}
		tr := trace(result.Position, end, ducked)
		if !tr.Valid {
			return result, false
		}
		// Match PM_StepSlideMove_: allsolid stops this branch, while
		// startsolid with a free endpoint may leave the overlap normally.
		// This predicts physics only; callers must separately certify safety.
		if tr.AllSolid {
			result.Velocity[2] = 0
			return result, true
		}
		if tr.Fraction > 0 {
			result.Segments = append(result.Segments, [2]Vec3{result.Position, tr.End})
			result.Position = tr.End
			planes = planes[:0]
		}
		if tr.Fraction == 1 {
			return result, true
		}
		remaining *= 1 - tr.Fraction
		result.Contacts = append(result.Contacts, tr.Normal)
		planes = append(planes, tr.Normal)
		accepted := false
		for i, normal := range planes {
			result.Velocity = clip(result.Velocity, normal)
			accepted = true
			for j, other := range planes {
				if j != i && dot(result.Velocity, other) < 0 {
					accepted = false
					break
				}
			}
			if accepted {
				break
			}
		}
		if !accepted {
			if len(planes) != 2 {
				result.Velocity = Vec3{}
				return result, true
			}
			a, b := planes[0], planes[1]
			direction := Vec3{a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0]}
			speed := dot(direction, result.Velocity)
			for axis := range direction {
				result.Velocity[axis] = direction[axis] * speed
			}
		}
		if dot(result.Velocity, velocity) <= 0 {
			result.Velocity = Vec3{}
			return result, true
		}
	}
	return result, true
}
